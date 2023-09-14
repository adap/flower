"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
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
    NDArrays,
    ndarrays_to_parameters,   # parameters_to_weights,
    parameters_to_ndarrays,   # weights_to_parameters,
)
# pylint: disable=no-member
# DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

DIR = '1E_up_theta_b_only_FedAvg+SWA_wo_moment'

# order classes by number of samples
def takeSecond(elem):
    return elem[1]

class SslClient(fl.client.NumPyClient):
    """Flower client implementing video SSL w/ PyTorch."""

    def __init__(self, model, train_dataset, test_dataset, cfg, args, distributed, logger, videossl):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.cfg = cfg
        self.distributed = distributed
        self.logger = logger
        self.videossl = videossl
        self.mu = 0.3

    def get_parameters(self, config):
        # Return local model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):

        state_dict = OrderedDict()
        params_dict = zip(self.model.state_dict().keys(), parameters)

        # define a new state_dict and update only the backbone weights but keep the cls weights
        for k, v in params_dict:
            if k.strip().split('.')[0] == 'backbone': # if there is any term backbone update its weights
                state_dict[k] = torch.from_numpy(v)
            else:
                state_dict[k] = self.model.state_dict()[k]  # keep the previous weights
    
        self.model.load_state_dict(state_dict, strict=True)
    
    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):

        global_round = int(config["epoch_global"])
        self.cfg.lr_config = dict(policy='step', step=[100, 200])

        chk_name_list = [fn for fn in os.listdir(self.cfg.work_dir) if fn.endswith('.pth')]
        chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
        
        # Gathering  local epochs if there are any
        if chk_epoch_list:
            chk_epoch_list.sort()
            checkpoint = os.path.join(self.cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')
            pr_model = torch.load(checkpoint)
            state_dict_pr = pr_model['state_dict']
            # load the model with previous state_dictionary
            self.model.load_state_dict(state_dict_pr, strict=True)
            
        # Update local model w/ global parameters
        self.set_parameters(parameters)
        
        self.videossl.train_model_cl(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.args,
            cfg=self.cfg,
            distributed=self.distributed,
            logger=self.logger
        )

        # Return updated model parameters to the server
        num_examples = len(self.train_dataset)  

        
        # fetch loss from log file
        work_dir = self.args.work_dir
        log_f_list = []
        for f in os.listdir(work_dir):
            if f.endswith('log.json'):
                num = int(''.join(f.split('.')[0].split('_')))
                log_f_list.append((f, num))

        # take the last log file
        log_f_list.sort(key=takeSecond)
        log_f_name = work_dir + '/' + log_f_list[-1][0]
        loss_list = []
        with open(log_f_name, 'r') as f:
            for line in f.readlines():
                line_dict = eval(line.strip())
                loss = float(line_dict['loss'])
                loss_list.append(loss)

        avg_loss = sum(loss_list) / len(loss_list)
        exp_loss = exp(- avg_loss)
        metrics = {'loss': exp_loss}

        # get the model keys 
        metrics['state_dict_key'] = [k for k in self.model.state_dict().keys()]

        return self.get_parameters(config=None), num_examples, metrics

    def evaluate(self, parameters, config):
        
        # for completion
        result = 0
       
        return float(0), int(1), {"accuracy": float(result)} # int(len(self.test_dataset)), 


def _temp_get_parameters(model):
    # Return local model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


