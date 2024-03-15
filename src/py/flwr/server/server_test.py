# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


from typing import List, Optional

import numpy as np

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Status,
    ndarray_to_bytes,
)
from flwr.server.client_manager import SimpleClientManager

from .client_proxy import ClientProxy
from .server import Server, evaluate_clients, fit_clients


class SuccessClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetPropertiesRes:
        """Raise an error because this method is not expected to be called."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetParametersRes:
        """Raise a error because this method is not expected to be called."""
        raise NotImplementedError()

    def fit(
        self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]
    ) -> FitRes:
        """Simulate fit by returning a success FitRes with simple set of weights."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[arr_serialized], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]
    ) -> EvaluateRes:
        """Simulate evaluate by returning a success EvaluateRes with loss 1.0."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )

    def reconnect(
        self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]
    ) -> DisconnectRes:
        """Simulate reconnect by returning a DisconnectRes with UNKNOWN reason."""
        return DisconnectRes(reason="UNKNOWN")


class FailingClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetPropertiesRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetParametersRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def fit(
        self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]
    ) -> FitRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]
    ) -> EvaluateRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def reconnect(
        self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]
    ) -> DisconnectRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()


def test_fit_clients() -> None:
    """Test fit_clients."""
    # Prepare
    clients: List[ClientProxy] = [
        FailingClient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: FitIns = FitIns(Parameters(tensors=[arr_serialized], tensor_type=""), {})
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = fit_clients(client_instructions, None, None, 0)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1].num_examples == 1


def test_eval_clients() -> None:
    """Test eval_clients."""
    # Prepare
    clients: List[ClientProxy] = [
        FailingClient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: EvaluateIns = EvaluateIns(
        Parameters(tensors=[arr_serialized], tensor_type=""),
        {},
    )
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = evaluate_clients(
        client_instructions=client_instructions,
        max_workers=None,
        timeout=None,
        group_id=0,
    )

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1].loss == 1.0
    assert results[0][1].num_examples == 1


def test_set_max_workers() -> None:
    """Test eval_clients."""
    # Prepare
    server = Server(client_manager=SimpleClientManager())

    # Execute
    server.set_max_workers(42)

    # Assert
    assert server.max_workers == 42
