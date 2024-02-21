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


import argparse
import csv
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

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
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_key_pairs,
    public_key_to_bytes,
)
from flwr.server.client_manager import SimpleClientManager

from .app import _try_setup_client_authentication
from .client_proxy import ClientProxy
from .server import Server, evaluate_clients, fit_clients


class SuccessClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        """Raise an error because this method is not expected to be called."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        """Raise a error because this method is not expected to be called."""
        raise NotImplementedError()

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        """Simulate fit by returning a success FitRes with simple set of weights."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[arr_serialized], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        """Simulate evaluate by returning a success EvaluateRes with loss 1.0."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        """Simulate reconnect by returning a DisconnectRes with UNKNOWN reason."""
        return DisconnectRes(reason="UNKNOWN")


class FailingClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
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
    results, failures = fit_clients(client_instructions, None, None)

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


def test_setup_client_auth() -> None:
    """Test setup client authentication."""
    # Generate keys
    _, first_public_key = generate_key_pairs()
    server_public_key = (
        b"ssh-ed25519 "
        b"AAAAC3NzaC1lZDI1NTE5AAAAIH2WQPMp+JHI9UxvrFuOphfZXN5CC12N3AKB6CjmRnpN"
    )
    server_private_key = b"""-----BEGIN OPENSSH PRIVATE KEY-----
    b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
    QyNTUxOQAAACB9lkDzKfiRyPVMb6xbjqYX2VzeQgtdjdwCgego5kZ6TQAAALBZcmCzWXJg
    swAAAAtzc2gtZWQyNTUxOQAAACB9lkDzKfiRyPVMb6xbjqYX2VzeQgtdjdwCgego5kZ6TQ
    AAAEDWjCVhWlskdnWQPyoRo6E/kwZBra82kIrH4P3UoZI9z32WQPMp+JHI9UxvrFuOphfZ
    XN5CC12N3AKB6CjmRnpNAAAAJ2RhbmllbG51Z3JhaGFARGFuaWVscy1NYWNCb29rLVByby
    5sb2NhbAECAwQFBg==
    -----END OPENSSH PRIVATE KEY-----"""
    _, second_public_key = generate_key_pairs()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize temporary files
        client_keys_file_path = Path(temp_dir) / "client_keys.csv"
        server_public_key_path = Path(temp_dir) / "server_public_key"
        server_private_key_path = Path(temp_dir) / "server_private_key"

        # Fill the files with relevant keys
        with open(client_keys_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    public_key_to_bytes(first_public_key).decode(),
                    public_key_to_bytes(second_public_key).decode(),
                ]
            )
        server_public_key_path.write_bytes(server_public_key)
        server_private_key_path.write_bytes(server_private_key)

        # Mock argparse with `require-client-authentication`` flag
        mock_args = argparse.Namespace(
            require_client_authentication=[
                str(client_keys_file_path),
                str(server_public_key_path),
                str(server_private_key_path),
            ]
        )

        expected_private_key = load_ssh_private_key(server_private_key, None)
        expected_public_key = load_ssh_public_key(server_public_key)

        # Run _try_setup_client_authentication
        result = _try_setup_client_authentication(mock_args)

        # Assert result with expected values
        assert result is not None
        assert result == (
            {
                public_key_to_bytes(first_public_key),
                public_key_to_bytes(second_public_key),
            },
            expected_public_key,
            expected_private_key,
        )
