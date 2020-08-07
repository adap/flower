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
"""Flower client app."""

import timeit
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from .client import Client


class KerasClient(ABC):
    """Abstract base class for Flower clients which use Keras."""

    def __init__(self, cid: str):
        self.cid = cid

    @abstractmethod
    def get_weights(self) -> Weights:
        """Return the current local model weights."""

    @abstractmethod
    def fit(self, weights: Weights, config: Dict[str, str]) -> Tuple[Weights, int, int]:
        """Refine the provided weights using the locally held dataset."""

    @abstractmethod
    def evaluate(
        self, weights: Weights, config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        """Evaluate the provided weights using the locally held dataset."""


class KerasClientWrapper(Client):
    """Wrapper which translates between Client and KerasClient."""

    def __init__(self, keras_client: KerasClient) -> None:
        super().__init__(cid=keras_client.cid)
        self.cid = keras_client.cid
        self.keras_client = keras_client

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        weights = self.keras_client.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        weights: Weights = parameters_to_weights(ins[0])
        config = ins[1]

        # Train
        fit_begin = timeit.default_timer()
        weights_prime, num_examples, num_examples_ceil = self.keras_client.fit(
            weights, config
        )
        fit_duration = timeit.default_timer() - fit_begin

        # Return FitRes
        parameters = weights_to_parameters(weights_prime)
        return parameters, num_examples, num_examples_ceil, fit_duration

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        # Deconstruct EvaluateIns
        weights: Weights = parameters_to_weights(ins[0])
        config = ins[1]

        # Evaluate and return
        num_examples, loss, accuracy = self.keras_client.evaluate(weights, config)
        return num_examples, loss, accuracy
