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

from flower.client import NetworkClient
from flower.client_manager import ClientManager
from flower.proto import transport_pb2_grpc
from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class ClientInfoMessageError(Exception):
    """Signifies the first message did not contain a ClientMessage.Info message."""


class ClientManagerRejectionError(Exception):
    """Signifies the client has been rejected by the client manager."""


def default_client_factory(cid: str, info: Dict[str, str]) -> NetworkClient:
    """Return NetworkClient instance."""
    return NetworkClient(cid=cid, info=info)


def register_client(
    client_manager: ClientManager, client: NetworkClient, context: grpc.ServicerContext
) -> None:
    """Try registering NetworkClient with ClientManager.
    If not successful set appropriate gRPC ServicerContext."""
    if not client_manager.register(client):
        raise ClientManagerRejectionError()

    def rpc_termination_callback():
        client_manager.unregister(client)

    context.add_callback(rpc_termination_callback)


def is_client_message_info(message: ClientMessage) -> None:
    """Check if message contains a ClientMessage.Info message"""
    if not message.HasField("info"):
        raise ClientInfoMessageError()


def is_not_client_message_info(message: ClientMessage) -> None:
    """Check if message contains other than ClientMessage.Info message"""
    if message.HasField("info"):
        raise ClientInfoMessageError()


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC instruction stream."""

    def __init__(
        self,
        client_manager: ClientManager,
        client_factory: Callable[
            [str, Dict[str, str]], NetworkClient
        ] = default_client_factory,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.client_factory: Callable[
            [str, Dict[str, str]], NetworkClient
        ] = client_factory

    def Join(  # pylint: disable=invalid-name
        self, request_iterator: Iterator[ClientMessage], context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        """Method will be invoked by each NetworkClient which participates in the network.

        Protocol:
            - The first ClientMessage has always have the connect field set
            - Subsequent messages should not have the connect field set
        """
        # A string identifying the peer that invoked the RPC being serviced.
        client_message_iterator = request_iterator
        peer = context.peer()

        yield ServerMessage(info=ServerMessage.GetClientInfo())

        try:
            client_message = next(client_message_iterator)
        except StopIteration:
            return

        try:
            is_client_message_info(client_message)
        except ClientInfoMessageError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "First message has to be a connect message!",
            )

        info = MessageToDict(client_message.info)
        client = self.client_factory(peer, info)

        try:
            register_client(self.client_manager, client, context)
        except ClientManagerRejectionError:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Client registeration failed!")

        # All subsequent messages will be pushed to client proxy directly
        for client_message in client_message_iterator:
            yield client.proxy.set_client_message_get_server_message(
                client_message=client_message
            )

        self.client_manager.unregister(client)
