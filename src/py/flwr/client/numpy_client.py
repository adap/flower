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
    Code,
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
    Status,
    parameters_to_weights,
    weights_to_parameters,
)

from .client import Client

DEPRECATION_WARNING_FIT = """
DEPRECATION WARNING: deprecated return format

    parameters, num_examples

move to

    parameters, num_examples, {"custom_key": custom_val}

instead. Note that the deprecated return format will be removed in a future
release.
"""
DEPRECATION_WARNING_EVALUATE_0 = """
DEPRECATION WARNING: deprecated return format

    num_examples, loss, accuracy

move to

    loss, num_examples, {"accuracy": accuracy}

instead. Note that the deprecated return format will be removed in a future
release.
"""
DEPRECATION_WARNING_EVALUATE_1 = """
DEPRECATION WARNING: deprecated return format

    num_examples, loss, accuracy, {"custom_key": custom_val}

move to

    loss, num_examples, {"accuracy": accuracy}

instead. Note that the deprecated return format will be removed in a future
release.
"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The return type should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

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
        PropertiesRes :
            Response containing `properties` of the client.
        """

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The local model parameters as a list of NumPy ndarrays.
        """

    @abstractmethod
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[List[np.ndarray], int, Dict[str, Scalar]],
        Tuple[List[np.ndarray], int],  # Deprecated
    ]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    @abstractmethod
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],  # Deprecated
        Tuple[int, float, float, Dict[str, Scalar]],  # Deprecated
    ]:
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) are still
        supported for compatibility reasons. They will however be removed
        in a future release, please migrate to (float, int, Dict[str, Scalar]).
        """


def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


class NumPyClientWrapper(Client):
    """Wrapper which translates between Client and NumPyClient."""

    def __init__(self, numpy_client: NumPyClient) -> None:
        self.numpy_client = numpy_client

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Return the current client properties."""
        properties = self.numpy_client.get_properties(ins.config)
        return PropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties,
        )

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
            print(DEPRECATION_WARNING_FIT)
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
            num_examples_ceil=num_examples,  # Deprecated
            fit_duration=fit_duration,  # Deprecated
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        results = self.numpy_client.evaluate(parameters, ins.config)
        if len(results) == 3:
            if (
                isinstance(results[0], float)
                and isinstance(results[1], int)
                and isinstance(results[2], dict)
            ):
                # Forward-compatible case: loss, num_examples, metrics
                results = cast(Tuple[float, int, Metrics], results)
                loss, num_examples, metrics = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    metrics=metrics,
                )
            elif (
                isinstance(results[0], int)
                and isinstance(results[1], float)
                and isinstance(results[2], float)
            ):
                # Legacy case: num_examples, loss, accuracy
                # This will be removed in a future release
                print(DEPRECATION_WARNING_EVALUATE_0)
                results = cast(Tuple[int, float, float], results)
                num_examples, loss, accuracy = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    accuracy=accuracy,  # Deprecated
                )
            else:
                raise Exception(
                    "Return value expected to be of type (float, int, dict)."
                )
        elif len(results) == 4:
            # Legacy case: num_examples, loss, accuracy, metrics
            # This will be removed in a future release
            print(DEPRECATION_WARNING_EVALUATE_1)
            results = cast(Tuple[int, float, float, Metrics], results)
            assert isinstance(results[0], int)
            assert isinstance(results[1], float)
            assert isinstance(results[2], float)
            assert isinstance(results[3], dict)
            num_examples, loss, accuracy, metrics = results
            evaluate_res = EvaluateRes(
                loss=loss,
                num_examples=num_examples,
                accuracy=accuracy,  # Deprecated
                metrics=metrics,
            )
        else:
            raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE)

        return evaluate_res
