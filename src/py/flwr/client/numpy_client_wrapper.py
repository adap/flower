# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Wrapper for NumPyClient objects."""

from typing import Callable, Dict

from flwr.client.typing import ClientLike
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Status,
)

from .client import Client
from .numpy_client import NumPyClient
from .numpy_client import has_evaluate as numpyclient_has_evaluate
from .numpy_client import has_fit as numpyclient_has_fit
from .numpy_client import has_get_parameters as numpyclient_has_get_parameters
from .numpy_client import has_get_properties as numpyclient_has_get_properties

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


def to_client(client_like: ClientLike) -> Client:
    """Take any Client-like object and return it as a Client."""
    if isinstance(client_like, NumPyClient):
        return _wrap_numpy_client(client=client_like)
    return client_like


def _constructor(self: Client, numpy_client: NumPyClient) -> None:
    self.numpy_client = numpy_client  # type: ignore


def _get_properties(self: Client, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Return the current client properties."""
    properties = self.numpy_client.get_properties(config=ins.config)  # type: ignore
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Success"),
        properties=properties,
    )


def _get_parameters(self: Client, ins: GetParametersIns) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_client.get_parameters(config=ins.config)  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
    )


def _evaluate(self: Client, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], float)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
    }

    # Add wrapper type methods (if overridden)

    if numpyclient_has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if numpyclient_has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if numpyclient_has_fit(client=client):
        member_dict["fit"] = _fit

    if numpyclient_has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore
