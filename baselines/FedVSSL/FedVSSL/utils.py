"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from CtP.pyvrl.apis import train_network, get_root_logger, set_random_seed, test_network 
from CtP.pyvrl.builder import build_model, build_dataset
from CtP.tools import train_net as  train_model_cl
# from CtP.tools import test_net as test_model_cl

# import _init_paths
import os
import re
import mmcv
import argparse
import shutil
from mmcv import Config
from mmcv.runner import init_dist
from mmcv.utils import collect_env
import pdb

def set_config_mmcv(args, cfg):
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    cfg.gpus = args.gpus
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # if the CLI args has already specified a resume_from path,
    # we will recover training process from it.
    # otherwise, if there exists a trained model in work directory, we will resume training from it
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    else:
        if os.path.isdir(cfg.work_dir):
            chk_name_list = [fn for fn in os.listdir(cfg.work_dir) if fn.endswith('.pth')]
            # if there may exists multiple checkpoint, we will select a latest one (with a highest epoch number)
            if len(chk_name_list) > 0:
                chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
                chk_epoch_list.sort()
                cfg.resume_from = os.path.join(cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')
                

    # setup data root directory
    if args.data_dir is not None:
        if 'train' in cfg.data:
            cfg.data.train.data_dir = args.data_dir
        if 'val' in cfg.data:
            cfg.data.val.data_dir = args.data_dir
        if 'test' in cfg.data:
            cfg.data.test.data_dir = args.data_dir

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config/save the config in the working directory
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.cfg)))

    # init logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    return distributed, logger

def load_model(args, cfg):
    
    """Load SSL model."""
    # set_config_mmcv(args, cfg)
    model = build_model(cfg.model)
    # TODO replace with videoSSL model
    return model

def load_data(args, cfg):
    """Load the data partition for a single client ID."""
    train_dataset = build_dataset(cfg.data.train)
    return train_dataset

def load_test_data(args, cfg):
    test_dataset = build_dataset(cfg.data.val)
    return test_dataset

def train_model_cl(model, train_dataset, args, cfg, distributed, logger):
    # model code
    train_network(model,
        train_dataset,
        cfg,
        distributed=distributed,
        logger=logger
    )
    
# def test_model_cl(model, test_dataset, args, cfg, distributed, logger):
#     result = test_network(model,
#         test_dataset,
#         cfg,
#         args,
#         distributed=distributed,
#         logger=logger
#     )
#     return result



