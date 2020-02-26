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
"""Provides contextmanager which manages a gRPC channel to connect to the server."""
from __future__ import annotations  # pylint: disable=unused-import

from contextlib import contextmanager
from queue import Queue
from typing import Callable, Iterator, Tuple

import grpc

from flower.grpc_server import DEFAULT_PORT, DEFAULT_SERVER_ADDRESS
from flower.proto.transport_pb2 import ClientMessage, ServerMessage
from flower.proto.transport_pb2_grpc import FlowerServiceStub

# Uncomment these flags in case you are debugging
# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "connectivity_state"


def on_channel_state_change(*args, **kwargs):
    """Print all args and kwargs."""
    print(*args, **kwargs)


@contextmanager
def insecure_grpc_connection(
    server_address: str = DEFAULT_SERVER_ADDRESS, port: int = DEFAULT_PORT
) -> Iterator[Tuple[Callable[[], ServerMessage], Callable[[ClientMessage], None]]]:
    """Establishe an insecure gRPC connection to a gRPC server."""
    channel = grpc.insecure_channel(f"{server_address}:{port}")

    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)  # type: ignore

    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    receive: Callable[[], ServerMessage] = lambda: next(server_message_iterator)
    send: Callable[[ClientMessage], None] = lambda msg: queue.put(msg, block=False)

    try:
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        print("Insecure gRPC channel closed")
