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
"""Tests for module server."""
import socket
import subprocess
from contextlib import closing
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Tuple, cast

from flwr.server.client_manager import SimpleClientManager
from flwr.server.grpc_server.grpc_server import start_grpc_server, valid_ssl_files

root_dir = dirname(abspath(join(__file__, "../../../../..")))


def load_certificates() -> Tuple[str, str, str]:
    """Generate and load SSL/TLS credentials/certificates.

    Utility function for loading for SSL/TLS enabled gRPC servertests.
    """
    # Trigger script which generates the certificates
    subprocess.run(["bash", "./dev/certificates/generate.sh"], check=True, cwd=root_dir)

    ssl_files = (
        join(root_dir, ".cache/certificates/ca.crt"),
        join(root_dir, ".cache/certificates/server.pem"),
        join(root_dir, ".cache/certificates/server.key"),
    )

    return ssl_files


def unused_tcp_port() -> int:
    """Return an unused port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


def test_valid_ssl_files_when_correct() -> None:
    """Test is validation function works correctly when passed valid list."""
    # Prepare
    ssl_files = (b"a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_ssl_files(ssl_files)

    # Assert
    assert is_valid


def test_valid_ssl_files_when_wrong() -> None:
    """Test is validation function works correctly when passed invalid list."""
    # Prepare
    ssl_files = ("not_a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_ssl_files(ssl_files)  # type: ignore

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

    ssl_files = load_certificates()

    # Execute
    server = start_grpc_server(
        client_manager=client_manager,
        server_address=f"[::]:{port}",
        ssl_files=(
            Path(ssl_files[0]).read_bytes(),
            Path(ssl_files[1]).read_bytes(),
            Path(ssl_files[2]).read_bytes(),
        ),
    )

    # Teardown
    server.stop(1)
