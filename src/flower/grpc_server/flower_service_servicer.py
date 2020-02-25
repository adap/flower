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
from flower.proto.transport_pb2 import ClientRequest, ServerResponse


class ConnectRequestError(Exception):
    """Signifies the first request did not contain a ClientRequest.Connect message."""


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


def is_connect_message(request: ClientRequest) -> None:
    """Check if message contains a ClientRequest.Connect message"""
    if not request.HasField("connect"):
        raise ConnectRequestError()


def is_not_connect_message(request: ClientRequest) -> None:
    """Check if message contains other than ClientRequest.Connect message"""
    if request.HasField("connect"):
        raise ConnectRequestError()


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
        self, request_iterator: Iterator[ClientRequest], context: grpc.ServicerContext
    ) -> Iterator[ServerResponse]:
        """Method will be invoked by each NetworkClient which participates in the network.

        Protocol:
            - The first ClientRequest has always have the connect field set
            - Subsequent messages should not have the connect field set
        """
        # A string identifying the peer that invoked the RPC being serviced.
        peer = context.peer()

        try:
            request = next(request_iterator)
        except StopIteration:
            return

        try:
            is_connect_message(request)
        except ConnectRequestError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "First message has to be a connect message!",
            )

        info = MessageToDict(request.connect.info)
        client = self.client_factory(peer, info)

        try:
            register_client(self.client_manager, client, context)
        except ClientManagerRejectionError:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Client registeration failed!")

        # Call get response with an empty ClientRequest as it will be discarded
        # and we have already processed it above.
        yield client.proxy.push_result_and_get_next_instruction(result=None)

        # All subsequent requests will be pushed to client proxy directly
        for request in request_iterator:
            try:
                is_not_connect_message(request)
            except ConnectRequestError:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "connect field only allowed in first message!",
                )
            yield client.proxy.push_result_and_get_next_instruction(result=request)

        self.client_manager.unregister(client)
