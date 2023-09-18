# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/8/10 21:40
import functools

import torch
# from models.network_swinir import SwinIR
# from models.network_spsr_v2_9_b_m_e_f import SPSRNetv2_9_b_m_e_f
from models.network_release import SPSRNet_release
# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def select_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    if net_type == 'SwinIR':
        netG = SwinIR(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            window_size=opt_net['window_size'],
            img_range=opt_net['img_range'],
            depths=opt_net['depths'],
            embed_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'],
            resi_connection=opt_net['resi_connection'],
        )
    else:
        try:
            netG = eval(net_type)(
                in_channel=opt_net["in_channel"],
                out_channel=opt_net['out_channel'],
                hidden_dim=opt_net['hidden_dim'],
                img_size=opt_net['img_size'],
                layer_num=opt_net["layer_num"],
                scale=opt_net["upscale"],
                window_size=opt_net['window_size'],
                norm_layer=opt_net["norm_layer"],
                upsample=opt_net["upsample"],
            )
        except Exception as e:
            raise NotImplementedError('netG [{:s}] is not found. Except {}'.format(net_type, e))

    print('Training model [{:s}] is created.'.format(net_type))
    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
# from models.network_unet import Discriminator_UNet


# def define_D(opt):
#     opt_net = opt['netD']
#     net_type = opt_net['net_type']
#
#     # ----------------------------------------
#     # discriminator_vgg_96
#     # ----------------------------------------
#
#     if net_type == 'discriminator_unet':
#         netD = Discriminator_UNet(input_nc=opt_net['in_nc'],
#                                   ndf=opt_net['base_nc'])
#     else:
#         raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))
#
#     # ----------------------------------------
#     # initialize weights
#     # ----------------------------------------
#     init_weights(netD,
#                  init_type=opt_net['init_type'],
#                  init_bn_type=opt_net['init_bn_type'],
#                  gain=opt_net['init_gain'])
#
#     return netD


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    '''

    :param net: model
    :param init_type:  pass init_weights type
        normal,uniform,xavier normal,xavier uniform,kaiming normal,kaiming uniform,orthogonal
    :param init_bn_type: pass init bn layer type
    :param gain: scaler
    :return:
    '''

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                torch.nn.init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    torch.nn.init.uniform_(m.weight.data, 0.1, 1.0)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    torch.nn.init.constant_(m.weight.data, 1.0)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        # 相当于遍历net的所有层来使用这个fn函数，不用单独写for了
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
