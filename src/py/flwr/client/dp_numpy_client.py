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
from flwr.common.typing import Config, NDArrays, Scalar


# Calculates the L2-norm of a potentially ragged array
def _get_update_norm(update: NDArrays) -> float:
    flattened_update = update[0]
    for i in range(1, len(update)):
        flattened_update = np.append(flattened_update, update[i])  # type: ignore
    return float(np.sqrt(np.sum(np.square(flattened_update))))


class DPNumPyClient(NumPyClient):
    """Wrapper for configuring a Flower client for DP."""

    def __init__(
        self, client: NumPyClient, adaptive_clip_enabled: bool = False
    ) -> None:
        super().__init__()
        self.client = client
        self.adaptive_clip_enabled = adaptive_clip_enabled

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

        # Calculating the factor to scale the update by
        update_norm = _get_update_norm(update)

        if "clip_norm" not in config:
            raise Exception("Clipping threshold not supplied by the server.")
        if not isinstance(config["clip_norm"], float):
            raise Exception("Clipping threshold should be a floating point value.")

        scaling_factor = min(1, config["clip_norm"] / update_norm)

        # Clipping update to bound sensitivity of aggregate at server
        update_clipped = [layer * scaling_factor for layer in update]

        if "noise_stddev" not in config:
            raise Exception("Scale of noise to be added not supplied by the server.")
        if not isinstance(config["noise_stddev"], float):
            raise Exception(
                "Scale of noise to be added should be a floating point value."
            )
        update_clipped_noised = [
            layer + np.random.normal(0, config["noise_stddev"], layer.shape)
            for layer in update_clipped
        ]

        for i, _ in enumerate(original_params):
            updated_params[i] = original_params[i] + update_clipped_noised[i]

        # Calculating value of norm indicator bit, required for adaptive clipping
        if self.adaptive_clip_enabled:
            metrics["norm_bit"] = not scaling_factor < 1

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)
