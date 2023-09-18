from torch.optim import Adam
from torch.optim import lr_scheduler

from losses import *
from models.model_base import ModelBase
from models.select_network import select_G
from utils import *


class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']  # training option
        self.scale = self.opt['scale']
        self.netG = select_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = select_G(opt).to(self.device).eval()
        self.G_lossfn = []
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        self.gradient = Get_gradient_nopadding()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='model')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='model')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label, save_best=False):
        temp_save_dir = self.save_dir
        if save_best is True:
            self.save_dir = os.path.join(self.save_dir, 'best')
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        # if self.opt_train['E_decay'] > 0:
        #     self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        self.save_dir = temp_save_dir

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']

        for lossfn_type in G_lossfn_type:
            if lossfn_type == 'l1':
                self.G_lossfn.append(nn.L1Loss().to(self.device))
            elif lossfn_type == 'mse':
                self.G_lossfn.append(nn.MSELoss().to(self.device))
            elif lossfn_type == 'l2sum':
                self.G_lossfn.append(nn.MSELoss(reduction='sum').to(self.device))
            elif lossfn_type == 'ssim':
                self.G_lossfn.append(SSIMLoss().to(self.device))
            elif lossfn_type == 'charbonnier':
                self.G_lossfn.append(CharbonnierLoss(
                    self.opt_train['G_charbonnier_eps'] if self.opt_train[
                                                               'G_charbonnier_eps'] is not None else 1e-3).to(
                    self.device))
            elif lossfn_type == 'perceptual':
                self.G_lossfn.append(PerceptualLoss().to(self.device))
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'StepLR':
            self.schedulers.append(lr_scheduler.StepLR(self.G_optimizer,
                                                       self.opt_train['G_scheduler_milestones'],
                                                       self.opt_train['G_scheduler_gamma']
                                                       ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                            self.opt_train['G_scheduler_periods'],
                                                                            self.opt_train[
                                                                                'G_scheduler_restart_weights'],
                                                                            self.opt_train['G_scheduler_eta_min']
                                                                            ))
        else:
            raise NotImplementedError

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def volume2frame(self, x):
        b, g, c, k = x.shape
        x = x.contiguous().view(-1, c, k).unsqueeze(1)
        return x

    def feed_data(self, data, need_H=True):
        self.L = []
        for d in data['L']:
            self.L.append(self.volume2frame(d).float().to(self.device))
        if need_H:
            self.H = []
            for d in data['H']:
                self.H.append(self.volume2frame(d).float().to(self.device))

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, x=None):
        if x is None:
            self.E = self.netG(self.L)
        else:
            self.E = self.netG(x)
        return self.E

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        # loss部分有时候还是要重写
        # logger.info('!!test')
        G_loss = 1e-8
        for i, func in enumerate(self.G_lossfn):
            for e, h in zip(self.E, self.H):
                # x4 E:[[120x120 240x240 iout] ,[...]] H:[hr , ...]
                # x2 E:240x240 iout
                G_loss += func(e, h)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'])

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.opt_train['tile'] is not None:
                tile = self.opt_train['tile']
                all_E = []
                all_L = []
                for L in self.L:
                    b, c, h_old, w_old = L.shape

                    if h_old % tile != 0:
                        h_pad = (h_old // tile + 1) * tile - h_old
                        w_pad = (w_old // tile + 1) * tile - w_old
                    else:
                        h_pad = w_pad = 0
                    # 对称填充
                    L = torch.cat([L, torch.flip(L, [2])], 2)[:, :, :h_old + h_pad, :]
                    L = torch.cat([L, torch.flip(L, [3])], 3)[:, :, :, :w_old + w_pad]
                    b, c, h, w = L.shape
                    h_idx_list = list(range(0, h - tile, tile)) + [h - tile]
                    w_idx_list = list(range(0, w - tile, tile)) + [w - tile]
                    E = torch.zeros(b, c, h * self.scale, w * self.scale).type_as(L)
                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = L[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                            self.netG_forward(in_patch)
                            E[..., h_idx * self.scale:(h_idx + tile) * self.scale,
                            w_idx * self.scale:(w_idx + tile) * self.scale].add_(E)
                    E = E[:, :, :h_old * self.scale, :w_old * self.scale]
                    L = L[:, :, :h_old, :w_old]
                    all_E.append(E)
                    all_L.append(L)
                self.E = all_E
                self.L = all_L
            else:
                self.netG_forward()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        '''
            可视化只显示第一个输入网络参数和第一个HR参数
        :param need_H: 是否需要 H
        :return: {'L': B,C,H,W ,'E': ,'H': }
        '''

        out_dict = OrderedDict()
        out_dict['L'] = self.L[0].detach().float().cpu()
        out_dict['E'] = self.E[0][-1].detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H[0].detach().float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
