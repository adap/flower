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
from typing import Callable, Iterator

import grpc

from flwr.proto import transport_pb2_grpc
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_manager import ClientManager
from flwr.server.grpc_server.grpc_bridge import GRPCBridge
from flwr.server.grpc_server.grpc_client_proxy import GrpcClientProxy


def default_bridge_factory() -> GRPCBridge:
    """Return GRPCBridge instance."""
    return GRPCBridge()


def default_grpc_client_factory(cid: str, bridge: GRPCBridge) -> GrpcClientProxy:
    """Return GrpcClientProxy instance."""
    return GrpcClientProxy(cid=cid, bridge=bridge)


def register_client(
    client_manager: ClientManager,
    client: GrpcClientProxy,
    context: grpc.ServicerContext,
) -> bool:
    """Try registering GrpcClientProxy with ClientManager."""
    is_success = client_manager.register(client)

    if is_success:

        def rpc_termination_callback() -> None:
            client.bridge.close()
            client_manager.unregister(client)

        context.add_callback(rpc_termination_callback)

    return is_success


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC message stream."""

    def __init__(
        self,
        client_manager: ClientManager,
        grpc_bridge_factory: Callable[[], GRPCBridge] = default_bridge_factory,
        grpc_client_factory: Callable[
            [str, GRPCBridge], GrpcClientProxy
        ] = default_grpc_client_factory,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.client_factory = grpc_client_factory

    def Join(  # pylint: disable=invalid-name
        self, request_iterator: Iterator[ClientMessage], context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        """Method will be invoked by each GrpcClientProxy which participates in
        the network.

        Protocol:
            - The first message is sent from the server to the client
            - Both ServerMessage and ClientMessage are message "wrappers"
                wrapping the actual message
            - The Join method is (pretty much) protocol unaware
        """
        peer = context.peer()
        bridge = self.grpc_bridge_factory()
        client = self.client_factory(peer, bridge)
        is_success = register_client(self.client_manager, client, context)

        if is_success:
            # Get iterators
            client_message_iterator = request_iterator
            server_message_iterator = bridge.server_message_iterator()

            # All messages will be pushed to client bridge directly
            while True:
                try:
                    # Get server message from bridge and yield it
                    server_message = next(server_message_iterator)
                    yield server_message
                    # Wait for client message
                    client_message = next(client_message_iterator)
                    bridge.set_client_message(client_message)
                except StopIteration:
                    break
