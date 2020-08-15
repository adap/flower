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


from logging import INFO

from flwr.common.logger import log

from .client import Client
from .grpc_client.connection import insecure_grpc_connection
from .grpc_client.message_handler import handle
from .keras_client import KerasClient, KerasClientWrapper


def start_client(server_address: str, client: Client) -> None:
    """Start a Flower Client which connects to a gRPC server."""
    with insecure_grpc_connection(server_address) as conn:
        receive, send = conn
        log(INFO, "Opened (insecure) gRPC connection")

        while True:
            server_message = receive()
            client_message = handle(client, server_message)
            send(client_message)


def start_keras_client(server_address: str, client: KerasClient) -> None:
    """Start a Flower KerasClient which connects to a gRPC server."""

    # Wrap the Keras client
    flower_client = KerasClientWrapper(client)

    # Start
    start_client(server_address, flower_client)
