import math

import torch
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F

from utils import PixelShuffleBlock, UpConvBlock, conv_block, get_gradient, Select
from einops import rearrange


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, in_channel, kernel_size=3, hidden_dim=32, stride=1, bias=True, pad_type='zero', norm_type=None,
                 act_type='gelu', mode='CNA'):
        super(ResidualDenseBlock5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(in_channel, hidden_dim, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(in_channel + hidden_dim, hidden_dim, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(in_channel + 2 * hidden_dim, hidden_dim, kernel_size, stride, bias=bias,
                                pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(in_channel + 3 * hidden_dim, hidden_dim, kernel_size, stride, bias=bias,
                                pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(in_channel + 4 * hidden_dim, in_channel, 3, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDBx(nn.Module):
    def __init__(self, in_channel, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='gelu', mode='CNA'):
        super(RRDBx, self).__init__()
        self.stack_num = stack_num
        self.RRDBx = nn.ModuleList([
            ResidualDenseBlock5C(in_channel, kernel_size, hidden_dim, stride, bias, pad_type, norm_type, act_type, mode) for _
            in range(self.stack_num)])

    def forward(self, x):
        for RRDB in self.RRDBx:
            x = RRDB(x)
        return x.mul(0.2) + x


class FeatureAdaption(nn.Module):
    def __init__(self, in_channel=32, use_residual=True, learnable=True):
        super(FeatureAdaption, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(in_channel, affine=False)
        if self.learnable:
            self.conv_1 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1, bias=True),
                                        nn.GELU())
            self.conv_gamma = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref_ori):  # lr (b,32,120,120)  ref (b,32,240,240)
        b, c, h, w = lr.shape
        lr_mean = torch.mean(lr.reshape(b, c, h * w), dim=-1, keepdim=True).reshape(b, c, 1, 1)
        lr_std = torch.std(lr.reshape(b, c, h * w), dim=-1, keepdim=True).reshape(b, c, 1, 1)
        ref_normed = self.norm_layer(ref_ori)  # b,32,120,120
        style = self.conv_1(torch.cat([lr, ref_ori], dim=1))  # b,32,120,120
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
        out = ref_normed * gamma + beta
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, C ,H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = rearrange(x, 'b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c', ws1=window_size, ws2=window_size)
    # windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    x = rearrange(windows, '(b h w) (w1 w2) c -> b c (h w1) (w w2)', h=H // window_size, w=W // window_size,
                  w1=window_size, w2=window_size)
    return x


def window_partition_downshuffle(x, window_size):
    """

    :param x: B,C,H,W
    :param window_size: window size
    :return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    h_interval, w_interval = int(H / window_size), int(W / window_size)
    y = []
    for i in range(h_interval):
        for j in range(w_interval):
            y.append(x[:, :, i::h_interval, j::w_interval])  # fold
    # windows = torch.stack(y, 2).contiguous().view(-1, window_size * window_size, C)
    y = torch.stack(y, 2)
    windows = rearrange(y, 'b c l w1 w2 -> (b l) (w1 w2) c', w1=window_size, w2=window_size)
    return windows


def window_reverse_downshuffle(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    scale = int(pow((H * W / window_size / window_size), 0.5))
    # x = windows.view(B, H // window_size * W // window_size, window_size, window_size, -1).permute(0, 4, 1, 2,
    #                                                                                                3).contiguous(
    # ).view(B, -1, window_size, window_size)

    x = rearrange(windows, '(b l) (w1 w2) c -> b (c l) w1 w2', b=B, w1=window_size, w2=window_size)
    pixshuffle = nn.PixelShuffle(scale)
    x = pixshuffle(x)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LAM_Module(nn.Module):
    """ Layer attention module"""

    def __init__(self):
        super(LAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, head, num, dim = x.size()
        x = x.permute(0, 3, 1, 2)
        proj_query = x.view(batchsize, dim, -1)
        proj_key = x.view(batchsize, dim, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = energy - torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        attention = self.softmax(energy)
        proj_value = x.view(batchsize, dim, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, dim, head, num)
        out = self.gamma * out + x
        out = out.permute(0, 2, 3, 1)
        return out


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.layer_att_other = LAM_Module()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if type(x) is tuple:
            x1, x2 = x
        else:
            raise NotImplementedError('{} is not tuple'.format(x))
        B_, N, C = x1.shape
        q = self.q(x1).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = self.layer_att_other(attn @ v).transpose(1, 2).reshape(B_, N, C) + q.transpose(1, 2).reshape(B_, N, C) + x1
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=5, shift_size=0, partition_type='window_partition',
                 mlp_ratio=4., scale=1, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.scale = scale
        self.shift_size = shift_size
        if partition_type == 'window_partition':
            self.window_partition = window_partition
            self.window_reverse = window_reverse
        else:
            self.window_partition = window_partition_downshuffle
            self.window_reverse = window_reverse_downshuffle

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.LCLG_1 = nn.Sequential(
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.GELU()
        )
        self.LCLG_2 = nn.Sequential(
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.GELU()
        )
        self.x1_pos_embedding = nn.Parameter(
            torch.ones(1, window_size * window_size, dim))
        self.x2_pos_embedding = nn.Parameter(
            torch.ones(1, window_size * window_size, dim))

        mlp_hidden_dim = int(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        img_mask = img_mask.permute(0, 3, 1, 2)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size*window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: list or tuple):
        if type(x) is not list and type(x) is not tuple:
            x1 = x2 = x
        else:
            x1, x2 = x
        B, C, H, W = x1.shape
        x_size = (H, W)
        x1 = self.LCLG_1(x1)
        x2 = self.LCLG_2(x2)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        # partition windows
        # B,C,h,w => B*num,window_size * window_size,C
        x1_windows = self.window_partition(shifted_x1, self.window_size)
        x2_windows = self.window_partition(shifted_x2, self.window_size)

        x1_windows = x1_windows + self.x1_pos_embedding
        x2_windows = x2_windows + self.x2_pos_embedding
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # nW*B, window_size*window_size, C
        if self.input_resolution == x_size:
            attn_windows = self.attn((x1_windows, x2_windows), mask=self.attn_mask)
        else:
            attn_windows = self.attn((x1_windows, x2_windows), mask=self.calculate_mask(x_size).to(x1.device))

        # FFN
        x = x1_windows + self.drop_path(attn_windows)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        shifted_x = self.window_reverse(x, self.window_size, H, W)  # B C, H W

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
        return x


class SLCC(nn.Module):
    """
        short long cross conv attention
    """

    def __init__(self, in_channel, img_size, hidden_dim, scale=2, window_size=5):
        super(SLCC, self).__init__()
        self.in_channel = in_channel
        self.scale = scale

        self.LG = RRDBx(in_channel=in_channel * 2, stack_num=1, kernel_size=3, hidden_dim=16)
        self.L_conv1_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.G_conv1_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.SWA = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=1,
                              window_size=window_size, shift_size=0, partition_type='window_partition')
        self.LWA = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=1,
                              window_size=window_size, shift_size=0,
                              partition_type='window_partition_downshuffle')
        self.IMA_1 = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=scale,
                                window_size=window_size, shift_size=0)
        self.IMA_2 = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=scale,
                                window_size=window_size, shift_size=window_size // 2)
        self.FA = FeatureAdaption(in_channel)
        self.EFF = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0),
                                 nn.GELU(),
                                 nn.Conv2d(in_channel, in_channel, 3, 1, 1))
        self.LNA = nn.Sequential(
            nn.InstanceNorm2d(in_channel),
            nn.GELU()
        )
        self.GNA = nn.Sequential(
            nn.InstanceNorm2d(in_channel),
            nn.GELU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ADM = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 2, bias=False),
        )
        self.conv_last = conv_block(in_channel, in_channel, kernel_size=1, stride=1)

    def forward(self, x):
        if type(x) is list:
            x, Fc_in = x

        fea = torch.cat([x, Fc_in], dim=1)
        B, C, H, W = x.shape
        LG = self.LG(fea).reshape(B, C, 2, H, W).permute(2, 0, 1, 3, 4)
        L, G = LG[0], LG[1]

        L_1_1 = self.SWA(L)
        L_out = self.LWA(L_1_1)

        fa = self.FA(G, Fc_in)

        G = self.EFF(torch.concat([fa, G], dim=1)) + G

        G_1_1 = self.IMA_1([G, fa])
        G_out = self.IMA_2([G_1_1, fa])

        L_out = self.LNA(L_out)
        G_out = self.GNA(G_out)
        y = self.avg_pool(x).view(B, C)
        y = self.ADM(y)

        ax = F.softmax(y, dim=1)

        out = L_out * ax[:, 0].view(B, 1, 1, 1) + G_out * ax[:, 1].view(B, 1, 1, 1)
        out = self.conv_last(out)
        out = out + x
        return out


class CohfT(nn.Module):
    def __init__(self, in_channel, img_size, hidden_dim, layer_num, scale=2, window_size=5, kernel_size=3,
                 stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA'):
        super(CohfT, self).__init__()
        self.RRDB1_1 = RRDBx(in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride,
                             bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.conv2_1 = conv_block(2 * in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.SLCC_block = nn.ModuleList(
            [SLCC(in_channel, img_size, hidden_dim, scale=scale, window_size=window_size) for _ in range(layer_num)])

        self.Select = Select(in_nc=in_channel, out_nc=hidden_dim)

    def forward(self, F_in, p_in, Fc_in, factor):
        E = self.RRDB1_1(F_in)
        Fs = self.conv1_1(E)
        B, C, H, W = p_in.shape
        Fs1 = torch.cat([p_in, Fs], dim=1)
        Fs2 = self.conv2_1(Fs1)
        res_Fs2 = Fs2
        for SLCC in self.SLCC_block:
            Fs2 = SLCC([Fs2, Fc_in])
        FL0 = res_Fs2 + Fs2[:, :C, :, :]
        s = self.Select(Fs, FL0)

        Fs_out = E + s
        Fs_out = F.interpolate(Fs_out, scale_factor=factor)
        FL0 = F.interpolate(FL0, scale_factor=factor)
        return Fs_out, FL0


class HRReconstruction(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, kernel_size=3, stride=1,
                 act_type=None, norm_type=None,
                 mode='CNA'):
        super(HRReconstruction, self).__init__()
        self.scale = scale
        self.RRDB1_1 = RRDBx(in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride, bias=True,
                             pad_type='zero',
                             norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=None, act_type=None)
        self.RRDB1_2 = RRDBx(2 * in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride,
                             bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_2 = conv_block(in_channel * 2, in_channel, kernel_size=kernel_size, norm_type=None,
                                  act_type=act_type)
        self.conv1_3 = conv_block(in_channel, out_channel, kernel_size=kernel_size, norm_type=None, act_type=None)
        # 第二条路
        self.conv2_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.conv2_4 = conv_block(in_channel, out_channel, kernel_size=1, norm_type=None, act_type=act_type)

    def forward(self, F_in, lr, P):
        Fc_in_sr = F.interpolate(lr, scale_factor=self.scale)
        F_in = self.RRDB1_1(F_in)
        F_in = self.conv1_1(F_in)

        P_ = self.conv2_1(P)
        P = P_ + P
        R_out = self.conv2_4(P)

        res = torch.cat([P, F_in], dim=1)
        I_out = self.RRDB1_2(res)
        I_out = self.conv1_2(I_out)
        I_out = self.conv1_3(I_out) + Fc_in_sr
        return I_out, R_out


class SPSRNetv2_9_b_m_e_c_p(nn.Module):
    same = False

    def __init__(self, img_size, in_channel=1, out_channel=1, hidden_dim=32, layer_num=3, scale=2, window_size=5,
                 norm_layer=None, act_type='gelu', mode='CNA', upsample=None):
        super(SPSRNetv2_9_b_m_e_c_p, self).__init__()
        self.scale = scale
        self.fea_conv = conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=norm_layer, act_type=act_type)
        self.Rs_grad_fea = nn.Sequential(
            conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=None, act_type=None),
            RRDBx(hidden_dim, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                  norm_type=norm_layer,
                  act_type=act_type, mode=mode)
        )

        self.Rc_grad_fea = nn.Sequential(
            conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=None, act_type=None),
            RRDBx(hidden_dim, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                  norm_type=norm_layer,
                  act_type=act_type, mode=mode),
        )

        if self.scale == 3:
            self.deep_feature_extract = nn.ModuleList([
                CohfT(hidden_dim, img_size, scale=3, hidden_dim=hidden_dim,
                      kernel_size=3, layer_num=layer_num, window_size=window_size,
                      stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA')]
            )
        else:
            num = self.scale // 2
            self.deep_feature_extract = nn.ModuleList([
                CohfT(hidden_dim, img_size * 2 ** i, scale=2, hidden_dim=hidden_dim,
                      kernel_size=3, layer_num=layer_num, window_size=window_size,
                      stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA')
                for i in range(num)])
        self.HR_Reconstruction = HRReconstruction(in_channel=hidden_dim, out_channel=out_channel, scale=scale,
                                                  act_type=act_type, norm_type=norm_layer, mode=mode)

        self.grad = get_gradient

    def forward(self, x: list):
        if type(x) is list:
            lr, Ref = x
        else:
            raise NotImplementedError('input must be two element and type is list but found {}'.format(type(x)))
        if lr.shape == Ref.shape:
            Ref = F.interpolate(Ref, size=(lr.shape[2] * self.scale, lr.shape[3] * self.scale))
        Rs = self.grad(lr)
        F0 = self.fea_conv(lr)
        P = self.Rs_grad_fea(Rs)
        Ref = self.Rc_grad_fea(Ref)
        F_in = F0.clone()
        B, C, H, W = F_in.shape
        for i, Cohf_T in enumerate(self.deep_feature_extract):
            if self.scale == 3:
                ref = F.interpolate(Ref, size=(H, W))
                factor = 3
            else:
                ref = F.interpolate(Ref, scale_factor=0.5 ** (self.scale // 2 - i))
                factor = 2
            F_in, P = Cohf_T(F_in, P, ref, factor)

        # output block
        I_out, R_out = self.HR_Reconstruction(F_in, lr, P)

        return [I_out, R_out]


if __name__ == "__main__":
    from thop import profile, clever_format

    device = 'cuda:0'
    net = SPSRNetv2_9_b_m_e_c_p(img_size=80, in_channel=1, out_channel=1, hidden_dim=32, scale=3, window_size=5,
                                layer_num=1).to(
        device)
    t2_gra = torch.randn(1, 1, 80, 80).to(device)
    t1_gra = torch.randn(1, 1, 240, 240).to(device)
    a = net([t2_gra, t1_gra])
    print(a[0].shape)
    print(a[1].shape)

    flops, params = profile(net, inputs=([t2_gra, t1_gra],))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)
