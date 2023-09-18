# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/8/11 20:48

from data.dataset_sr import *

def select_data(opt):
    dataset_type = opt['dataset_type']
    if dataset_type == 'DatasetBrainSRT':
        datasets = DatasetBrainSRT(opt)
    elif dataset_type == 'DatasetIXIT':
        datasets = DatasetIXIT(opt)
    else:
        datasets = eval(dataset_type)(opt)

    return datasets
