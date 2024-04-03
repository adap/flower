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
"""Servicer for FlowerService.

Relevant knowledge for reading this modules code:
- https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
"""

import itertools
import uuid
from typing import Callable, Iterator, cast, Optional

import grpc
from iterators import TimeoutIterator

from flwr.proto import transport_pb2_grpc  # pylint: disable=E0611
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.common.aws import BucketManager
from flwr.server.client_manager import ClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    ResWrapper,
)
from flwr.server.superlink.fleet.grpc_bidi.grpc_client_proxy import GrpcClientProxy
from py.flwr.common import serde


def default_bridge_factory() -> GrpcBridge:
    """Return GrpcBridge instance."""
    return GrpcBridge()


def default_grpc_client_proxy_factory(cid: str, bridge: GrpcBridge, bucket_manager: Optional[BucketManager]) -> GrpcClientProxy:
    """Return GrpcClientProxy instance."""
    return GrpcClientProxy(cid=cid, bridge=bridge, bucket_manager=bucket_manager)


def register_client_proxy(
    client_manager: ClientManager,
    client_proxy: GrpcClientProxy,
    context: grpc.ServicerContext,
) -> bool:
    """Try registering GrpcClientProxy with ClientManager."""
    is_success = client_manager.register(client_proxy)
    if is_success:

        def rpc_termination_callback() -> None:
            client_proxy.bridge.close()
            client_manager.unregister(client_proxy)

        context.add_callback(rpc_termination_callback)
    return is_success


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC message stream."""

    def __init__(
        self,
        client_manager: ClientManager,
        grpc_bridge_factory: Callable[[], GrpcBridge] = default_bridge_factory,
        grpc_client_proxy_factory: Callable[
            [str, GrpcBridge, Optional[BucketManager]], GrpcClientProxy
        ] = default_grpc_client_proxy_factory,
        bucket_manger: Optional[BucketManager] = None,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.client_proxy_factory = grpc_client_proxy_factory
        self.bucket_manager = bucket_manger

    def Join(  # pylint: disable=invalid-name
        self,
        request_iterator: Iterator[ClientMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        """Facilitate bi-directional streaming of messages between server and client.

        Invoked by each gRPC client which participates in the network.

        Protocol:
        - The first message is sent from the server to the client
        - Both `ServerMessage` and `ClientMessage` are message "wrappers"
          wrapping the actual message
        - The `Join` method is (pretty much) unaware of the protocol
        """
        # When running Flower behind a proxy, the peer can be the same for
        # different clients, so instead of `cid: str = context.peer()` we
        # use a `UUID4` that is unique.
        cid: str = uuid.uuid4().hex
        bridge = self.grpc_bridge_factory()
        client_proxy = self.client_proxy_factory(cid, bridge, self.bucket_manager)
        is_success = register_client_proxy(self.client_manager, client_proxy, context)

        if is_success:
            # Get iterators
            client_message_iterator = TimeoutIterator(
                iterator=request_iterator, reset_on_next=True
            )       
            ins_wrapper_iterator = bridge.ins_wrapper_iterator()

            # All messages will be pushed to client bridge directly
            while True:
                try:
                    # Get ins_wrapper from bridge and yield server_message
                    ins_wrapper = next(ins_wrapper_iterator)
                    timeout = ins_wrapper.timeout

                    for server_msg in ins_wrapper.raw_message_stream():
                        yield server_msg
                    

                    # Set current timeout, might be None
                    if timeout is not None:
                        client_message_iterator.set_timeout(timeout)

                    try:
                        def read_until_stream_end():
                            for client_message in client_message_iterator:
                                if client_message is client_message_iterator.get_sentinel():
                                    raise TimeoutError
                                client_message = cast(ClientMessage, client_message)
                                yield client_message
                                if serde.is_client_message_end(client_message):
                                    break

                        res_wrapper = ResWrapper(read_until_stream_end())
                        bridge.set_res_wrapper(res_wrapper)

                    except TimeoutError:
                        # Important: calling `context.abort` in gRPC always
                        # raises an exception so that all code after the call to
                        # `context.abort` will not run. If subsequent code should
                        # be executed, the `rpc_termination_callback` can be used
                        # (as shown in the `register_client` function).
                        details = f"Timeout of {timeout}sec was exceeded."
                        context.abort(
                            code=grpc.StatusCode.DEADLINE_EXCEEDED,
                            details=details,
                        )
                        # This return statement is only for the linter so it understands
                        # that client_message in subsequent lines is not None
                        # It does not understand that `context.abort` will terminate
                        # this execution context by raising an exception.
                        return
                except StopIteration:
                    break
                    # Wait for client message
