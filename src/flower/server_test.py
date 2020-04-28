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
"""Flower server tests."""


from typing import List

import numpy as np

from flower.strategy.parameter import ndarray_to_bytes

from .client_proxy import ClientProxy
from .server import evaluate_clients, fit_clients
from .typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, ParametersRes


class SuccessClient(ClientProxy):
    """Test class."""

    def get_parameters(self) -> ParametersRes:
        # This method is not expected to be called
        raise Exception()

    def fit(self, ins: FitIns) -> FitRes:
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        return Parameters(tensors=[arr_serialized], tensor_type=""), 1

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return 1, 1.0


class FailingCLient(ClientProxy):
    """Test class."""

    def get_parameters(self) -> ParametersRes:
        raise Exception()

    def fit(self, ins: FitIns) -> FitRes:
        raise Exception()

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise Exception()


def test_fit_clients() -> None:
    """Test fit_clients."""
    # Prepare
    clients: List[ClientProxy] = [
        FailingCLient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: FitIns = (Parameters(tensors=[arr_serialized], tensor_type=""), {})
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = fit_clients(client_instructions)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1] == 1


def test_eval_clients() -> None:
    """Test eval_clients."""
    # Prepare
    clients: List[ClientProxy] = [
        FailingCLient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: EvaluateIns = (Parameters(tensors=[arr_serialized], tensor_type=""), {})
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = evaluate_clients(client_instructions)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][0] == 1
    assert results[0][1] == 1.0
