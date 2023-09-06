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
import mmcv
from mmcv import Config
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
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from strategy import FedVSSL

if __name__ == "__main__":
    import _init_paths
    import utilis 

    os.chdir("/fedssl/")
    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 2, "num_gpus": 1}  # each client will get allocated 1 CPUs
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    base_work_dir = '/k400_' + DIR
    rounds = 1

    # configure the strategy
    strategy = FedVSSL(
        fraction_fit=0.05,
        fraction_eval=0.02,
        min_fit_clients=5,
        min_eval_clients=1,
        min_available_clients=pool_size,
        on_fit_config_fn=fit_config,
    )


    def main(cid: str):
        # Parse command line argument `cid` (client ID)
        #        os.environ["CUDA_VISIBLE_DEVICES"] = cid
        args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl = initial_setup(cid, base_work_dir, rounds)
        return SslClient(model, train_dataset, test_dataset, cfg, args, distributed, logger, utils)


    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=main,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        ray_init_args=ray_config,
    )

def initial_setup(cid, base_work_dir, rounds, light=False):
    import _init_paths
    import utils
    cid_plus_one = str(int(cid) + 1)
    args = Namespace(
        cfg='path to the configuration file.py',
        checkpoint=None, cid=int(cid), data_dir='/DATA', gpus=1,
        launcher='none',  
        local_rank=0, progress=False, resume_from=None, rounds=6, seed=7, validate=False,
        work_dir=base_work_dir + '/client' + cid_plus_one)
    
    print("Starting client", args.cid)
    cfg = Config.fromfile(args.cfg)
    cfg.total_epochs = 1  ### Used for debugging. Comment to let config file set number of epochs
    cfg.data.train.data_source.ann_file = 'DATA/Kinetics-400_annotations/client_dist' + cid_plus_one + '.json'
    
    
    distributed, logger = videossl.set_config_mmcv(args, cfg)
    
    # load the model
    model = videossl.load_model(args, cfg)
    # load the training data
    
    train_dataset = utils.load_data(args, cfg)
    
    # load the test data
    test_dataset = utils.load_test_data(args, cfg)
    
    return args, cfg, distributed, logger, model, train_dataset, test_dataset, videossl
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
    # save_path = HydraConfig.get().runtime.output_dir
# 