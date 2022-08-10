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

import numpy as np

from flwr.client.client import Client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


# Calculates the L2-norm of a potentially ragged array
def _get_update_norm(update):  # type: ignore
    flattened_layers = []
    for layer in update:
        flattened_layers.append(layer.ravel())
    flattened_update = np.concatenate(flattened_layers)  # type: ignore
    return np.linalg.norm(flattened_update)  # type: ignore


class DPClient(Client):
    """Wrapper for configuring a Flower client for DP."""

    def __init__(self, client: Client, adaptive_clip_enabled: bool = False) -> None:
        super().__init__()
        self.client = client
        self.adaptive_clip_enabled = adaptive_clip_enabled

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return self.client.get_properties(ins)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:

        # Global model received by the wrapped client at the beginning of the round
        original_ndarrays = copy.deepcopy(parameters_to_ndarrays(ins.parameters))

        # Getting the updated model from the wrapped client
        res = self.client.fit(ins)
        updated_ndarrays = parameters_to_ndarrays(res.parameters)

        # Update = updated model - original model
        update = [x - y for (x, y) in zip(updated_ndarrays, original_ndarrays)]

        # Calculating the factor to scale the update by
        update_norm = _get_update_norm(update)  # type: ignore
        scaling_factor = min(1, ins.config["clip_norm"] / update_norm)

        # Clipping update to bound sensitivity of aggregate at server
        update_clipped = [layer * scaling_factor for layer in update]  # type: ignore

        update_clipped_noised = [
            layer + np.random.normal(0, ins.config["noise_stddev"], layer.shape)
            for layer in update_clipped
        ]
        res.parameters = ndarrays_to_parameters(
            [x + y for (x, y) in zip(original_ndarrays, update_clipped_noised)]
        )

        # Calculating value of norm indicator bit, required for adaptive clipping
        if self.adaptive_clip_enabled:
            res.metrics["norm_bit"] = not scaling_factor < 1

        return res

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)
