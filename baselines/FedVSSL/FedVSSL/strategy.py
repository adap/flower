"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
# import all all the necessary libraries
import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import os
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner.checkpoint import load_state_dict, get_state_dict, save_checkpoint
import re
import mmcv
import time
import shutil
import time
import shutil
from flwr.common import parameter
import pdb # for debugging
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

DIR = '1E_up_theta_b_only_FedAvg+SWA_wo_moment'


class FedVSSL(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds: int=1, mix_coeff:float=0.2, swbeta:int=0,  *args, **kwargs):

        assert isinstance(swbeta, int) and swbeta >= 0 and swbeta <= 1,  "the value must be an integer and either 0 or 1 "
        self.num_rounds = num_rounds,
        self.mix_coeff = mix_coeff # coefficient for mixing the loss and fedavg aggregation methods
        self.swbeta = swbeta    # 0: SWA off; 1:SWA on 
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Divide the weights based on the backbone and classification head
        ######################################################################################################3

        # Aggregate all the weights and the number of examples 
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        # Aggregate all the weights and the loss
        loss_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['loss'])
            for client, fit_res in results
        ]


         # aggregate the weights of the backbone 
        weights_loss = aggregate(loss_results) # loss-based
        weights_avg = aggregate(weight_results) # Fedavg

        
        weights: NDArrays = [v * self.mix_coeff for v in weights_avg] + [(1 - self.mix_coeff) * v for v in weights_loss] # Equation 3 in FedVSSL paper
        
        # # aggregate the weights of the backbone
        # weights = aggregate(weight_results) # loss-based

        # create a directory to save the global checkpoints
        glb_dir = 'ucf_' + DIR
        mmcv.mkdir_or_exist(os.path.abspath(glb_dir))

        

        # load the previous weights if there are any

        if server_round > 1 and self.swbeta == 1:
            chk_name_list = [fn for fn in os.listdir(glb_dir) if fn.endswith('.npz')]
            chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('round')]
            if chk_epoch_list:
                chk_epoch_list.sort()
                print(chk_epoch_list)
                # select the most recent epoch
                checkpoint = os.path.join(glb_dir, f'round-{chk_epoch_list[-1]}-weights.array.npz')
                # load the previous model weights
                params = np.load(checkpoint, allow_pickle=True)
                params = params['arr_0'].item()
                print("The weights has been loaded")
                params = parameters_to_ndarrays(params) # return a list
                weights_avg = [np.asarray((0.5 * B + 0.5 * A)) for A, B in zip(weights, params)] # perform SWA
                weights_avg = ndarrays_to_parameters(weights_avg)
        else:
            print("The results are saved without performing SWA")
            weights_avg = ndarrays_to_parameters(weights)

        if weights_avg is not None:
            # save weights
            print(f"round-{server_round}-weights...",)
            np.savez(os.path.join(glb_dir, f"round-{server_round}-weights.array"), weights_avg)

        return weights_avg, {}


def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# class FedVSSL(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ):
#         """Aggregate fit results using weighted average."""
#         if not results:
#             return None, {}
#         # Do not aggregate if there are failures and failures are not accepted
#         if not self.accept_failures and failures:
#             return None, {}
#         # Convert results
#         weights_results_loss = [
#             (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['loss'])
#             for client, fit_res in results
#         ]
#
#         weights_results_fedavg = [
#             (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#             for client, fit_res in results
#         ]
#
#         weights_loss = aggregate(weights_results_loss)
#         weights_fedavg = aggregate(weights_results_fedavg)
#         alpha = 0.7
#         weights = alpha * weights_loss + (1 - alpha) * weights_fedavg
#
#         glb_dir = 'ucf_' + DIR
#         mmcv.mkdir_or_exist(os.path.abspath(glb_dir))
#
#         # load the previous weights if there are any
#         if rnd > 1:
#             chk_name_list = [fn for fn in os.listdir(glb_dir) if fn.endswith('.npz')]
#             chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('round')]
#             if chk_epoch_list:
#                 chk_epoch_list.sort()
#                 print(chk_epoch_list)
#                 # select the most recent epoch
#                 checkpoint = os.path.join(glb_dir, f'round-{chk_epoch_list[-1]}-weights.array.npz')
#                 # load the previous model weights
#                 params = np.load(checkpoint, allow_pickle=True)
#                 params = params['arr_0']
#                 print("The weights have been loaded")
#                 weights = 0.5 * weights + 0.5 * params
#
#         if weights is not None:
#             # save weights
#             print(f"round-{rnd}-weights...",)
#             np.savez(os.path.join(glb_dir, f"round-{rnd}-weights.array"), weights)
#
#         return parameters_to_ndarrays(weights), {}
