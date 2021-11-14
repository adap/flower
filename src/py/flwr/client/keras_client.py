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
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    ParametersRes,
    Properties,
    PropertiesIns,
    PropertiesRes,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from .client import Client


class KerasClient(ABC):
    """Abstract base class for Flower clients which use Keras."""

    @abstractmethod
    def get_weights(self) -> Weights:
        """Return the current local model weights.

        Returns:
            The local model weights as a list of NumPy ndarrays. In many cases,
            it will be sufficient to just return the return value of Keras'
            `model.get_weights()`.
        """

    @abstractmethod
    def get_properties(self, config: Config) -> Properties:
        """Returns a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        PropertiesRes:
            Response containing `properties` of the client.
        """

    @abstractmethod
    def fit(
        self, weights: Weights, config: Dict[str, Scalar]
    ) -> Union[Tuple[Weights, int, int], Tuple[Weights, int, int, Metrics]]:
        """Refine/train the provided weights using the locally held dataset.

        Arguments:
            weights: List[numpy.ndarray]. The current (global) model weights.
                This argument has the structure expected by Keras'
                `model.set_weights(weights)` and returned by Keras'
                `model.get_weights()`.
            config: Dict[str, Scalar]. Configuration parameters which allow the
                server to influence training on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to set the number of (local) training epochs.

        Returns:
            weights: Weights. The locally updated model weights, usually
                obtained by calling Keras' `model.get_weights()`.
            num_examples (int): The number of examples used for training.
            num_examples_ceil (int): The maximum number of examples that might
                have been used during training.  If the client does not
                terminate training early (e.g., due to a timeout or other
                stopping condition), then `num_examples == num_examples_ceil`.
            metrics (Metrics, optional): A dictionary mapping arbitrary string
                keys to values of type bool, bytes, float, int, or str. Metrics
                can be used to communicate arbitrary values back to the server.
        """

    @abstractmethod
    def evaluate(
        self, weights: Weights, config: Dict[str, Scalar]
    ) -> Union[Tuple[int, float, float], Tuple[int, float, float, Metrics]]:
        """Evaluate the provided weights using the locally held dataset.

        Arguments:
            weights: List[numpy.ndarray]. The current (global) model weights.
                This argument has the structure expected by Keras'
                `model.set_weights(weights)` and returned by Keras'
                `model.get_weights()`.
            config: Dict[str, Scalar]. Configuration parameters which allow the
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


class KerasClientWrapper(Client):
    """Wrapper which translates between Client and KerasClient."""

    def __init__(self, keras_client: KerasClient) -> None:
        self.keras_client = keras_client

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        weights = self.keras_client.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        properties = self.keras_client.get_properties(ins.config)
        return PropertiesRes(properties=properties)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        weights: Weights = parameters_to_weights(ins.parameters)

        # Train
        fit_begin = timeit.default_timer()
        results = self.keras_client.fit(weights, ins.config)
        if len(results) == 3:
            results = cast(Tuple[List[np.ndarray], int, int], results)
            weights_prime, num_examples, num_examples_ceil = results
            metrics: Optional[Metrics] = None
        elif len(results) == 4:
            results = cast(Tuple[List[np.ndarray], int, int, Metrics], results)
            weights_prime, num_examples, num_examples_ceil, metrics = results

        # Return FitRes
        fit_duration = timeit.default_timer() - fit_begin
        weights_prime_proto = weights_to_parameters(weights_prime)
        return FitRes(
            parameters=weights_prime_proto,
            num_examples=num_examples,
            num_examples_ceil=num_examples_ceil,
            fit_duration=fit_duration,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        weights: Weights = parameters_to_weights(ins.parameters)

        results = self.keras_client.evaluate(weights, ins.config)
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
