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
import json
from contextlib import contextmanager
from logging import DEBUG
from queue import Queue
from typing import Callable, Iterator, Tuple

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
    service_config_json = json.dumps(
        {
            "methodConfig": [
                {
                    # To apply retry to all methods, put [{}] in the "name" field
                    "name": [{}],
                    "retryPolicy": {
                        "maxAttempts": 8,
                        "initialBackoff": "0.1s",
                        "maxBackoff": "1s",
                        "backoffMultiplier": 2,
                        "retryableStatusCodes": ["UNAVAILABLE"],
                    },
                }
            ]
        }
    )
    options = []
    # NOTE: the retry feature will be enabled by default >=v1.40.0
    options.append(("grpc.enable_retries", 1))
    options.append(("grpc.service_config", service_config_json))
    options.append(("grpc.max_send_message_length", max_message_length))
    options.append(("grpc.max_receive_message_length", max_message_length))

    channel = grpc.insecure_channel(server_address, options=options)
    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)

    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    receive: Callable[[], ServerMessage] = lambda: next(server_message_iterator)
    send: Callable[[ClientMessage], None] = lambda msg: queue.put(msg, block=False)

    try:
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "Insecure gRPC channel closed")
