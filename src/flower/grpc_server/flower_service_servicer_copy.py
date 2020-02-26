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
"""Servicer for FlowerService.

Relevant knowledge for reading this modules code:
    - https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
"""
from typing import Callable, Dict, Iterator

import grpc
from google.protobuf.json_format import MessageToDict

from flower.client_manager import ClientManager
from flower.grpc_server.grpc_proxy_client import GRPCProxyClient
from flower.proto import transport_pb2_grpc
from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class ClientInfoMessageError(Exception):
    """Signifies the first message did not contain a ClientMessage.Info message."""


class ClientManagerRejectionError(Exception):
    """Signifies the client has been rejected by the client manager."""


def default_client_factory(cid: str) -> GRPCProxyClient:
    """Return NetworkClient instance."""
    return GRPCProxyClient(cid=cid)


def register_client(
    client_manager: ClientManager,
    client: GRPCProxyClient,
    context: grpc.ServicerContext,
) -> None:
    """Try registering NetworkClient with ClientManager.
    If not successful raise Exception."""
    if not client_manager.register(client):
        raise ClientManagerRejectionError()

    def rpc_termination_callback():
        client.bridge.close()
        client_manager.unregister(client)

    context.add_callback(rpc_termination_callback)

class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC instruction stream."""

    def __init__(
        self,
        client_manager: ClientManager,
        client_factory: Callable[
            [str, Dict[str, str]], GRPCProxyClient
        ] = default_client_factory,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.client_factory: Callable[
            [str, Dict[str, str]], GRPCProxyClient
        ] = client_factory

    def Join(  # pylint: disable=invalid-name
        self, request_iterator: Iterator[ClientMessage], context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        """Method will be invoked by each NetworkClient which participates in the network.

        Protocol:
            - The first ClientMessage has always have the connect field set
            - Subsequent messages should not have the connect field set
        """
        peer = context.peer()
        client = self.client_factory(cid=peer)
        register_client(self.client_manager, client, context)

        # Get iterators
        client_message_iterator = request_iterator
        server_message_iterator = iter(client.bridge.get_server_message, None)
        
        # All subsequent messages will be pushed to client bridge directly
        while True:
            try:
                # Get server message from bridge and yield it
                server_message = next(server_message_iterator)
                yield server_message
                # Wait for client message
                client_message = next(client_message_iterator)
                client.bridge.set_client_message(client_message)
            except StopIteration:
                break

