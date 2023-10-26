"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed

import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import os
import flwr as fl
import numpy as np
from math import exp
import torch
import torch.nn as nn
# pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
import mmcv
# from mmcv import Config
from mmengine.config import Config
from mmcv.runner.checkpoint import load_state_dict, get_state_dict, save_checkpoint
import re
import ray
import time
import shutil
from flwr.common import parameter
from functools import reduce
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,   # parameters_to_weight in the original FedVSSL repository
    parameters_to_ndarrays,   # weights_to_parameters in the original FedVSSL repository
)
# import CtP.tools._init_paths
from .strategy import FedVSSL
from .client import SslClient

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

def t_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition mayb

def initial_setup(cid, base_work_dir, rounds, data_dir, num_gpus, partition_dir):
    import utils as utils
    cid_plus_one = str(int(cid) + 1)
    args = Namespace(
        cfg='fedvssl/conf/mmcv_conf/r3d_18_ucf101/pretraining.py', # Path to the pretraining configuration file
        checkpoint=None, cid=int(cid), data_dir=data_dir, gpus=num_gpus,
        launcher='none',
        local_rank=0, progress=False, resume_from=None, rounds=rounds, seed=7, validate=False,
        work_dir=base_work_dir + '/client' + cid_plus_one)

    print("Starting client", args.cid)
    # fetch the configuration file
    cfg = Config.fromfile(args.cfg) 
    # define the client data files; usually contains paths to the samples and annotations
    cfg.data.train.data_source.ann_file = partition_dir + '/client_dist' + cid_plus_one + '.json' 

    distributed, logger = utils.set_config_mmcv(args, cfg)
    # These two arguments needs to be false during federated pretraining otherwise the client will load the previously saved checkpoint

    cfg.resume_from = False
    cfg.load_from = False

    # load the model
    model = utils.load_model(args, cfg)
    # load the training data
    train_dataset = utils.load_data(args, cfg)

    # since during pretraining of FedVSSL we don't need any testing data, we can left it empty.
    test_dataset = " " 

    return args, cfg, distributed, logger, model, train_dataset, test_dataset, utils

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    if cfg.pre_training:
        # first the paths needs to be defined otherwise the program may not be able to locate the files of the ctp
        from .utils import init_p_paths
        init_p_paths("fedvssl")

        client_resources = {"num_cpus": cfg.cpus_per_client, "num_gpus": cfg.gpus_per_client}
        base_work_dir = cfg.exp_name
        rounds = cfg.rounds
        data_dir = cfg.data_dir
        partition_dir = cfg.partition_dir
        num_gpus = cfg.gpus_per_client

        def client_fn(cid: str):
            args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl = initial_setup(cid,
                                                                                                         base_work_dir,
                                                                                                         rounds,
                                                                                                         data_dir,
                                                                                                         num_gpus,
                                                                                                         partition_dir)
            return SslClient(model, train_dataset, test_dataset, cfg, args, distributed, logger, videossl)

        # configure the strategy
        strategy = instantiate(
            cfg.strategy,
            on_fit_config_fn=fit_config,
        )

         # (optional) specify ray config
        ray_config = {"include_dashboard": cfg.include_dashboard}

        # start simulation
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.pool_size,
            client_resources=client_resources,
            config=fl.server.ServerConfig(num_rounds=cfg.rounds),
            strategy=strategy,
            ray_init_args=ray_config,
        )
    else:
        import subprocess
        import os
        import textwrap
        from mmcv.runner import load_state_dict
        import textwrap
        from mmcv.runner import load_state_dict
        # import FedVSSL.CtP
        # from FedVSSL.CtP.configs.ctp.r3d_18_kinetics import finetune_ucf101
        from .CtP.pyvrl.builder import build_model, build_dataset

        # we give an example on how one can perform fine-tuning uisng UCF-101 dataset.
        cfg_path = "fedvssl/CtP/configs/ctp/r3d_18_kinetics/finetune_ucf101.py"
        cfg_ = Config.fromfile(cfg_path)
        cfg_.model.backbone['pretrained'] = None

        # build a model using the configuration file from Ctp repository
        model = build_model(cfg_.model)

        # path to the pretrained model. We provide certain federated pretrained model that can be easily downloaded
        # from the following link: https://github.com/yasar-rehman/FEDVSSL
        # here we gave an example with FedVSSL (alpha=0, beta=0) checkpoint file
        # The files after federated pretraining are usually saved in .npz format.

        pretrained = cfg.pretrained_model_path

        # conversion of the .npz files to the .pth format. If the files are saved in .npz format
        if pretrained.endswith('.npz'):
            # following changes are made here
            params = np.load(pretrained, allow_pickle=True)
            params = params['arr_0'].item()
            params = parameters_to_ndarrays(params)
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = {
                'state_dict': OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            }
            torch.save(state_dict, './model_pretrained.pth')

        # -----------------------------------------------------------------------------------------------------------------------
        # The cfg_path need to be updated with the following updated configuration contents to be able to load the pretrained model.
        # Instead of executing the blow mentioned code, one can also directly modify the "pretrained" variable by opening the path represented
        # by the config_path variable
        #
        config_content = textwrap.dedent('''\
                _base_ = ['../../recognizers/_base_/model_r3d18.py',
                '../../recognizers/_base_/runtime_ucf101.py']
                work_dir = './output/ctp/r3d_18_kinetics/finetune_ucf101/'
                model = dict(
                    backbone=dict(
                        pretrained='./model_pretrained.pth', 
                    ),
                )
               ''').strip("\n")

        with open(cfg_path, 'w') as f:
            f.write(config_content)

        process_obj = subprocess.run(["bash", "fedvssl/CtP/tools/dist_train.sh",
                                      f"{cfg_path}", "4",
                                      f"--work_dir {cfg.exp_name_finetune}",
                                      f"--data_dir {cfg.data_dir}"])

        # -----------------------------------------------------------------------------------------------------------------------
        # The cfg_path need to be updated with the following updated configuration contents to be able to load the pretrained model.
        # Instead of executing the blow mentioned code, one can also directly modify the "pretrained" variable by opening the file represented
        # by the cfg_path_test variable
        #
        config_content_test = textwrap.dedent('''\
                _base_ = ['../../recognizers/_base_/model_r3d18.py',
                '../../recognizers/_base_/runtime_ucf101.py']
                work_dir = './output/ctp/r3d_18_ucf101/finetune_ucf101/'
                model = dict(
                    backbone=dict(
                    pretrained='/finetune/ucf101/epoch_150.pth',
                ),
                )
               ''').strip("\n")

        cfg_path_test = "fedvssl/CtP/configs/ctp/r3d_18_ucf101/finetune_ucf101.py"
        with open(cfg_path_test, 'w') as f:
            f.write(config_content_test)

        # Evaluating the finetuned model
        process_obj = subprocess.run(["bash", "fedvssl/CtP/tools/dist_test.sh",
                                      f"{cfg_path_test}", "4",
                                      f"--work_dir {cfg.exp_name_finetune}",
                                      f"--data_dir {cfg.data_dir}", 
                                      f"--progress"])


if __name__ == "__main__":
    main()
