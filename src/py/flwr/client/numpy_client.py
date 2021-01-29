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
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    ParametersRes,
    Scalar,
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
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[Tuple[List[np.ndarray], int], Tuple[List[np.ndarray], int, Metrics]]:
        """Train the provided parameters using the locally held dataset.

        Arguments:
            parameters: List[numpy.ndarray]. The current (global) model
                parameters.
            config: Dict[str, Scalar]. Configuration parameters which allow the
                server to influence training on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to set the number of (local) training epochs.

        Returns:
            parameters: List[numpy.ndarray]. The locally updated model
                parameters.
            num_examples (int): The number of examples used for training.
            metrics (Metrics, optional): A dictionary mapping arbitrary string
                keys to values of type bool, bytes, float, int, or str. Metrics
                can be used to communicate arbitrary values back to the server.
        """

    @abstractmethod
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[Tuple[int, float, float], Tuple[int, float, float, Metrics]]:
        """Evaluate the provided weights using the locally held dataset.

        Args:
            parameters (List[np.ndarray]): The current (global) model
                parameters.
            config (Dict[str, Scalar]): Configuration parameters which allow the
                server to influence evaluation on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to influence the number of examples used for
                evaluation.

        Returns:
            num_examples (int): The number of examples used for evaluation.
            loss (float): The evaluation loss of the model on the local
                dataset.
            accuracy (float, deprecated): The accuracy of the model on the
                local test dataset.
            metrics (Metrics, optional): A dictionary mapping arbitrary string
                keys to values of type bool, bytes, float, int, or str. Metrics
                can be used to communicate arbitrary values back to the server.
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
        results = self.numpy_client.fit(parameters, ins.config)
        if len(results) == 2:
            results = cast(Tuple[List[np.ndarray], int], results)
            parameters_prime, num_examples = results
            metrics: Optional[Metrics] = None
        elif len(results) == 3:
            results = cast(Tuple[List[np.ndarray], int, Metrics], results)
            parameters_prime, num_examples, metrics = results

        # Return FitRes
        fit_duration = timeit.default_timer() - fit_begin
        parameters_prime_proto = weights_to_parameters(parameters_prime)
        return FitRes(
            parameters=parameters_prime_proto,
            num_examples=num_examples,
            num_examples_ceil=num_examples,  # num_examples == num_examples_ceil
            fit_duration=fit_duration,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        results = self.numpy_client.evaluate(parameters, ins.config)
        # Note that accuracy is deprecated and will be removed in a future release
        if len(results) == 3:
            results = cast(Tuple[int, float, float], results)
            num_examples, loss, accuracy = results
            metrics: Optional[Metrics] = None
        elif len(results) == 4:
            results = cast(Tuple[int, float, float, Metrics], results)
            num_examples, loss, accuracy, metrics = results
        return EvaluateRes(
            num_examples=num_examples, loss=loss, accuracy=accuracy, metrics=metrics
        )
