import argparse
import math
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.select_datasets import select_data
from models import select_Model
from utils import mkdir, mkdirs, tensor2single, get_logger, psnr, ssim, get_dist_info, init_dist
from utils import utils_option as option

# def main(json_path='options/x3/train_braint_spsr_s3_d32_w5_n1.json'):
def main(json_path='options/x3/train_ixit_spsr_v2_9_b_m_e_f_s3_d32_w5_n1.json'):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--gpu_id', default=None, type=int)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    para = parser.parse_args()
    opt['dist'] = para.dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
        mkdir(os.path.join(opt['path']['models'], 'best'))
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    if init_path_G != None:
        opt['path']['pretrained_netG'] = init_path_G
    if init_path_E != None:
        opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')

    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    assert opt['scale'] == opt['netG']["upscale"], 'Error! scale is need be the same'
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        logger = get_logger(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['train']['manual_seed'] = seed

    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = select_data(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=False,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=False,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = select_data(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'eval':
            eval_set = select_data(dataset_opt)
            eval_loader = DataLoader(eval_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------

    model = select_Model(opt)
    model.init_train()

    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    print('start train')
    best_save_psnr = 22
    for epoch in range(4000):
        if current_step > 10 * 10000:
            break

        if opt['dist']:
            train_sampler.set_epoch(epoch)
        bar = tqdm(train_loader)
        if opt['train']['checkpoint_test'] == 0:
            opt['train']['checkpoint_test'] = len(bar) + 1
        for i, train_data in enumerate(bar):
            current_step += 1

            # -------------------------------
            #  update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            # -------------------------------
            #  feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            # -------------------------------
            #  optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            #  training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0 and current_step > 20000:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = []
                avg_ssim = []
                model.netG.eval()
                if current_step < 90000:
                    try:
                        loader = eval_loader
                    except:
                        loader = test_loader
                else:
                    try:
                        loader = test_loader
                    except:
                        loader = eval_loader
                for idx, test_data in enumerate(loader):

                    image_name_ext = os.path.basename(test_data['path'][0])
                    result = []
                    model.feed_data(test_data)
                    maxn = min(model.L[0].shape[0], opt['maxn'])
                    minn = opt['minn']
                    n_channels = 1
                    for j in range(minn, maxn, n_channels):
                        with torch.no_grad():
                            lr = model.L[0][j:min(j + n_channels, maxn), :, :, :]
                            hr = model.L[1][j:min(j + n_channels, maxn), :, :, :]
                            model.netG_forward([lr, hr])
                            visuals = model.current_visuals()
                            E_img = tensor2single(visuals['E'])
                            # E_img = visuals['E'].squeeze().float().cpu().numpy()

                            result.append(E_img)

                    result = np.array(result)
                    current_psnr = psnr(result, test_data['H'][0].squeeze().cpu().numpy()[minn:maxn])
                    current_ssim = ssim(result, test_data['H'][0].squeeze().cpu().numpy()[minn:maxn])
                    logger.info('{:->4d}--> {:>10s} | {:<4.4f}dB {:<4.4f}'.format(idx, image_name_ext, current_psnr,
                                                                                  current_ssim))
                    avg_psnr.append(current_psnr)
                    avg_ssim.append(current_ssim)

                # testing log
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.4f}dB|{:.4f}\n'.format(epoch, current_step,
                                                                                          np.mean(avg_psnr),
                                                                                          np.mean(avg_ssim)))
                if np.mean(avg_psnr) >= best_save_psnr:
                    best_save_psnr = np.mean(avg_psnr)
                    model.save(current_step, save_best=True)

                model.netG.train()


if __name__ == '__main__':
    main()
