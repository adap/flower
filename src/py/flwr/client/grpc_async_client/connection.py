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
"""Provides contextmanager which manages a gRPC channel to connect to the
server."""
from contextlib import contextmanager
from logging import DEBUG
from typing import Callable, Dict, Iterator, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.proto.transport_pb2_grpc import FlowerServiceStub

# Uncomment these flags in case you are debugging
# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "connectivity_state"


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


@contextmanager
def insecure_grpc_connection(
    server_address: str, max_message_length: int = GRPC_MAX_MESSAGE_LENGTH
) -> Iterator[Tuple[Callable[[], ServerMessage], Callable[[ClientMessage], None]]]:
    """Establish an insecure gRPC connection to a gRPC server."""
    channel = grpc.insecure_channel(
        server_address,
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    channel.subscribe(on_channel_state_change)
    stub = FlowerServiceStub(channel)

    # Use a dict to be able to access it in the
    # subsequent two functions by reference
    storage: Dict[str, ServerMessage] = {"server_message": stub.Async(ClientMessage())}

    def receive() -> ServerMessage:
        return storage["server_message"]

    def send(msg: ClientMessage) -> None:
        if storage["server_message"] is not None:
            msg.reply_to = storage["server_message"].identifier
        storage["server_message"] = stub.Async(msg)

    try:
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "Insecure gRPC channel closed")
