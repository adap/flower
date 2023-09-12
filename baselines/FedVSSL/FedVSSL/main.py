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
    ndarrays_to_parameters,   # parameters_to_weights,
    parameters_to_ndarrays,   # weights_to_parameters,
)
import CtP.tools._init_paths
from strategy import FedVSSL
from client import SslClient


DIR = '1E_up_theta_b_only_FedAvg+SWA_wo_moment'

def initial_setup(cid, base_work_dir, rounds, light=False):
    import utils
    cid_plus_one = str(int(cid) + 1)
    args = Namespace(
        cfg='conf/mmcv_conf/r3d_18_ucf101/pretraining.py',
        checkpoint=None, cid=int(cid), data_dir='/local/scratch/ucf101', gpus=1,
        launcher='none',
        local_rank=0, progress=False, resume_from=None, rounds=6, seed=7, validate=False,
        work_dir=base_work_dir + '/client' + cid_plus_one)

    print("Starting client", args.cid)
    cfg = Config.fromfile(args.cfg)
    cfg.total_epochs = 1  ### Used for debugging. Comment to let config file set number of epochs
    cfg.data.train.data_source.ann_file = 'non_iid/client_dist' + cid_plus_one + '.json'

    distributed, logger = utils.set_config_mmcv(args, cfg)

    # load the model
    model = utils.load_model(args, cfg)
    # load the training data

    train_dataset = utils.load_data(args, cfg)

    # load the test data
    test_dataset = utils.load_test_data(args, cfg)

    return args, cfg, distributed, logger, model, train_dataset, test_dataset, utils

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config


if __name__ == "__main__":
    train_flag = True
    # train_flag = False
    if train_flag:
        pool_size = 2  # number of dataset partions (= number of total clients)
        client_resources = {"num_cpus": 2, "num_gpus": 1}  # each client will get allocated 1 CPUs
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        base_work_dir = 'ucf_' + DIR
        rounds = 1

        def main(cid: str):
            # Parse command line argument `cid` (client ID)
            #        os.environ["CUDA_VISIBLE_DEVICES"] = cid
            args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl = initial_setup(cid, base_work_dir, rounds)
            return SslClient(model, train_dataset, test_dataset, cfg, args, distributed, logger, videossl)

        # configure the strategy
        strategy = FedVSSL(
            fraction_fit=1,
            # fraction_eval=0,
            min_fit_clients=2,
            # min_eval_clients=0,
            min_available_clients=pool_size,
            on_fit_config_fn=fit_config,
        )
         # (optional) specify ray config
        ray_config = {"include_dashboard": False}

        # start simulation
        hist = fl.simulation.start_simulation(
            client_fn=main,
            num_clients=pool_size,
            client_resources=client_resources,
            config=fl.server.ServerConfig(num_rounds=rounds),
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
        import CtP
        from CtP.configs.ctp.r3d_18_kinetics import finetune_ucf101
        from CtP.pyvrl.builder import build_model, build_dataset
        
        # we give an example on how one can perform fine-tuning uisng UCF-101 dataset. 
        cfg_path = "CtP/configs/ctp/r3d_18_kinetics/finetune_ucf101.py" 
        cfg = Config.fromfile(cfg_path)
        cfg.model.backbone['pretrained'] = None
        
        # build a model using the configuration file from Ctp repository
        model = build_model(cfg.model)

        # path to the pretrained model. We provide certain federated pretrained model that can be easily downloaded 
        # from the following link: https://github.com/yasar-rehman/FEDVSSL
        # here we gave an exampe with FedVSSL (alpha=0, beta=0) checkpoint file
        # The files after federated pretraining are usually saved in .npz format. 
        
        pretrained = "/home/data1/round-540-weights.array.npz"
        
        # conversion of the .npz files to the .pth format. If the files are saved in .npz format
        if pretrained.endswith('.npz'):
            # following changes are made here
            params = np.load(pretrained, allow_pickle=True)
            params = params['arr_0'].item()
            params = parameters_to_ndarrays(params)
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = {
                'state_dict':OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            }
            torch.save(state_dict, './model_pretrained.pth')
        
       
    #-----------------------------------------------------------------------------------------------------------------------
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

        # start the finetuning with ucf-101.
        from CtP.pyvrl.builder import build_model, build_dataset
        
        # we give an example on how one can perform fine-tuning uisng UCF-101 dataset. 
        cfg_path = "CtP/configs/ctp/r3d_18_kinetics/finetune_ucf101.py" 
        cfg = Config.fromfile(cfg_path)
        cfg.model.backbone['pretrained'] = None
        
        # build a model using the configuration file from Ctp repository
        model = build_model(cfg.model)

        # path to the pretrained model. We provide certain federated pretrained model that can be easily downloaded 
        # from the following link: https://github.com/yasar-rehman/FEDVSSL
        # here we gave an exampe with FedVSSL (alpha=0, beta=0) checkpoint file
        # The files after federated pretraining are usually saved in .npz format. 
        
        pretrained = "/home/data1/round-540-weights.array.npz"
        
        # conversion of the .npz files to the .pth format. If the files are saved in .npz format
        if pretrained.endswith('.npz'):
            # following changes are made here
            params = np.load(pretrained, allow_pickle=True)
            params = params['arr_0'].item()
            params = parameters_to_ndarrays(params)
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = {
                'state_dict':OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            }
            torch.save(state_dict, './model_pretrained.pth')
        
       
    #-----------------------------------------------------------------------------------------------------------------------
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

        # start the finetuning with ucf-101.
        process_obj = subprocess.run(["bash", "CtP/tools/dist_train.sh",\
        "CtP/configs/ctp/r3d_18_kinetics/finetune_ucf101.py", "4",\
        f"--work_dir /finetune/ucf101/",
        f"--data_dir /DATA",\
        f"--pretrained /path to the pretrained checkpoint",\
        f"--validate"])



# import hydra
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def main(cfg: DictConfig) -> None:
#     """Run the baseline.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """
#     # 1. Print parsed config
#     print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
#
