# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for configuring a Flower client for DP."""


from abc import ABC, abstractmethod
from flwr.common.typing import Weights
from time import process_time
import numpy as np
import copy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    parameters_to_weights,
    weights_to_parameters
)
from flwr.client.client import Client


class DPClientTimed(Client):
    """Wrapper for configuring a Flower client for DP."""
    def __init__(self, client:Client, adaptive_clip_enabled:bool = False) -> None:
        super().__init__()
        self.client = client
        self.adaptive_clip_enabled = adaptive_clip_enabled

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.client.get_properties(ins)
    
    def get_parameters(self) -> ParametersRes:
        return self.client.get_parameters()
    
    # Calculates the L2-norm of a potentially ragged array 
    def __get_update_norm(self, update:Weights):
        flattened_update= []
        for layer in update:
            flattened_update.append(layer.ravel())
        return np.linalg.norm(np.concatenate(flattened_update))

    def fit(self, ins: FitIns) -> FitRes:
        
        # Global model received by the wrapped client at the beginning of the round
        original_weights = copy.deepcopy(parameters_to_weights(ins.parameters))

        # Getting the updated model from the wrapped client
        
        res = self.client.fit(ins)
    
        start_time = process_time()
        updated_weights = parameters_to_weights(res.parameters)

        # Update = updated model - original model
        update = [x-y for (x,y) in zip(updated_weights, original_weights)]
        
        # Calculating the factor to scale the update by
        update_norm = self.__get_update_norm(update)
        scaling_factor = min(1,ins.config["clip_norm"]/update_norm)

        # Clipping update to bound sensitivity of aggregate at server
        update_clipped = [layer*scaling_factor for layer in update]
       
        # Adding noise to the clipped update
        update_clipped_noised = [layer + np.random.normal(0, ins.config["noise_stddev"], layer.shape) for layer in update_clipped]
        res.parameters = weights_to_parameters([x+y for (x,y) in zip(original_weights, update_clipped_noised)])
        
        # Calculating the value of the norm indicator bit, required for adaptive clipping
        if self.adaptive_clip_enabled:
             res.metrics["norm_bit"] = False if scaling_factor < 1 else True
        
        end_time = process_time()
        elapsed_time_fit = end_time - start_time
        res.metrics["time_fit"] = res.metrics["time_fit"] + elapsed_time_fit
        return res

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)
