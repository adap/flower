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
from typing import Dict, List, Tuple

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    parameters_to_weights,
    weights_to_parameters,
)

from .client import Client


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters.

        Returns:
            The local model parameters as a list of NumPy ndarrays.
        """

    @abstractmethod
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        """Train the provided parameters using the locally held dataset.

        Arguments:
            parameters: List[numpy.ndarray]. The current (global) model
                parameters.
            config: Dict[str, str]. Configuration parameters which allow the
                server to influence training on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to set the number of (local) training epochs.

        Returns:
            A tuple containing two elements: Updated parameters and an `int`
            representing the number of examples used for training.
        """

    @abstractmethod
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        """Evaluate the provided weights using the locally held dataset.

        Arguments:
            parameters: List[numpy.ndarray]. The current (global) model
                parameters.
            config: Dict[str, str]. Configuration parameters which allow the
                server to influence evaluation on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to influence the number of examples used for
                evaluation.

        Returns:
            A tuple containing three elements: An `int` representing the number
            of examples used for evaluation, a `float` representing the loss,
            and a `float` representing the accuracy of the (global) model
            weights on the local dataset.
        """


class NumPyClientWrapper(Client):
    """Wrapper which translates between Client and NumPyClient."""

    def __init__(self, numpy_client: NumPyClient) -> None:
        self.numpy_client = numpy_client

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        parameters = self.numpy_client.get_parameters()
        parameters_proto = weights_to_parameters(parameters)
        return ParametersRes(parameters=parameters_proto)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        # Train
        fit_begin = timeit.default_timer()
        parameters_prime, num_examples = self.numpy_client.fit(parameters, ins.config)
        fit_duration = timeit.default_timer() - fit_begin

        # Return FitRes
        parameters_prime_proto = weights_to_parameters(parameters_prime)
        return FitRes(
            parameters=parameters_prime_proto,
            num_examples=num_examples,
            num_examples_ceil=num_examples,  # num_examples == num_examples_ceil
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)
        num_examples, loss, accuracy = self.numpy_client.evaluate(
            parameters, ins.config
        )
        return EvaluateRes(num_examples=num_examples, loss=loss, accuracy=accuracy)
