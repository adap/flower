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
from contextlib import closing
from typing import cast

from flwr.server.client_manager import SimpleClientManager
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server


def unused_tcp_port() -> int:
    """Return an unused port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


def test_integration_start_and_shutdown_server() -> None:
    """Create server and check if FlowerServiceServicer is returned."""
    # Prepare
    port = unused_tcp_port()
    client_manager = SimpleClientManager()

    # Execute
    server = start_insecure_grpc_server(
        client_manager=client_manager, server_address=f"[::]:{port}"
    )

    # Teardown
    server.stop(1)
