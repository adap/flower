# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for module server."""


import socket
import subprocess
from contextlib import closing
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Tuple, cast

from flwr.server.client_manager import SimpleClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import (
    start_grpc_server,
    valid_certificates,
)

root_dir = dirname(abspath(join(__file__, "../../../../../../..")))


def load_certificates() -> Tuple[str, str, str]:
    """Generate and load SSL credentials/certificates.

    Utility function for loading for SSL-enabled gRPC servertests.
    """
    # Trigger script which generates the certificates
    subprocess.run(["bash", "./dev/certificates/generate.sh"], check=True, cwd=root_dir)

    certificates = (
        join(root_dir, ".cache/certificates/ca.crt"),
        join(root_dir, ".cache/certificates/server.pem"),
        join(root_dir, ".cache/certificates/server.key"),
    )

    return certificates


def unused_tcp_port() -> int:
    """Return an unused port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


def test_valid_certificates_when_correct() -> None:
    """Test is validation function works correctly when passed valid list."""
    # Prepare
    certificates = (b"a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)

    # Assert
    assert is_valid


def test_valid_certificates_when_wrong() -> None:
    """Test is validation function works correctly when passed invalid list."""
    # Prepare
    certificates = ("not_a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)  # type: ignore

    # Assert
    assert not is_valid


def test_integration_start_and_shutdown_insecure_server() -> None:
    """Create server and check if FlowerServiceServicer is returned."""
    # Prepare
    port = unused_tcp_port()
    client_manager = SimpleClientManager()

    # Execute
    server = start_grpc_server(
        client_manager=client_manager, server_address=f"[::]:{port}"
    )

    # Teardown
    server.stop(1)


def test_integration_start_and_shutdown_secure_server() -> None:
    """Create server and check if FlowerServiceServicer is returned."""
    # Prepare
    port = unused_tcp_port()
    client_manager = SimpleClientManager()

    certificates = load_certificates()

    # Execute
    server = start_grpc_server(
        client_manager=client_manager,
        server_address=f"[::]:{port}",
        certificates=(
            Path(certificates[0]).read_bytes(),
            Path(certificates[1]).read_bytes(),
            Path(certificates[2]).read_bytes(),
        ),
    )

    # Teardown
    server.stop(1)
