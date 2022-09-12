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


import copy
from typing import Dict, Tuple

import numpy as np

from flwr.client.numpy_client import NumPyClient
from flwr.common.dp import add_gaussian_noise, clip_by_l2
from flwr.common.typing import Config, NDArrays, Scalar


class DPFedAvgNumPyClient(NumPyClient):
    """Wrapper for configuring a Flower client for DP."""

    def __init__(self, client: NumPyClient) -> None:
        super().__init__()
        self.client = client

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        original_params = copy.deepcopy(parameters)
        # Getting the updated model from the wrapped client
        updated_params, num_examples, metrics = self.client.fit(parameters, config)

        # Update = updated model - original model
        update = [np.subtract(x, y) for (x, y) in zip(updated_params, original_params)]

        if "dpfedavg_clip_norm" not in config:
            raise Exception("Clipping threshold not supplied by the server.")
        if not isinstance(config["dpfedavg_clip_norm"], float):
            raise Exception("Clipping threshold should be a floating point value.")

        # Clipping
        update, clipped = clip_by_l2(update, config["dpfedavg_clip_norm"])

        if "dpfedavg_noise_stddev" in config:
            if not isinstance(config["dpfedavg_noise_stddev"], float):
                raise Exception(
                    "Scale of noise to be added should be a floating point value."
                )
            # Noising
            update = add_gaussian_noise(update, config["dpfedavg_noise_stddev"])

        for i, _ in enumerate(original_params):
            updated_params[i] = original_params[i] + update[i]

        # Calculating value of norm indicator bit, required for adaptive clipping
        if "dpfedavg_adaptive_clip_enabled" in config:
            if not isinstance(config["dpfedavg_adaptive_clip_enabled"], bool):
                raise Exception(
                    "dpfedavg_adaptive_clip_enabled should be a boolean-valued flag."
                )
            metrics["dpfedavg_norm_bit"] = not clipped

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)
