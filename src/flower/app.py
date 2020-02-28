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
"""Flower App."""
from typing import Dict

from flower.client import Client
from flower.grpc_client.connection import insecure_grpc_connection
from flower.grpc_client.message_handler import handle
from flower.grpc_server.grpc_server import start_insecure_grpc_server
from flower.server import Server


def start_server(server: Server, config: Dict[str, int]) -> None:
    """Start a Flower server using the gRPC transport layer."""
    grpc_server = start_insecure_grpc_server(client_manager=server.client_manager())

    # Fit model
    hist = server.fit(num_rounds=config["num_rounds"])
    print(hist)

    # Evaluate the final trained model
    loss = server.evaluate()
    print(f"Final loss after training: {loss}")

    # Stop the gRPC server
    grpc_server.stop(1)


def start_client(client: Client) -> None:
    """Start a Flower client which connects to a gRPC server."""
    with insecure_grpc_connection() as conn:
        receive, send = conn

        while True:
            server_message = receive()
            client_message = handle(client, server_message)
            send(client_message)
