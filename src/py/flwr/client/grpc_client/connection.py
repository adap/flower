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
"""Contextmanager for a gRPC streaming channel to the Flower server."""


import uuid
from contextlib import contextmanager
from logging import DEBUG
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator, Optional, Tuple, Union

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.proto.transport_pb2_grpc import FlowerServiceStub

# The following flags can be uncommented for debugging. Other possible values:
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
# import os
# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "tcp,http"


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


@contextmanager
def grpc_connection(
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
) -> Iterator[
    Tuple[
        Callable[[], Optional[TaskIns]],
        Callable[[TaskRes], None],
        Optional[Callable[[], None]],
        Optional[Callable[[], None]],
    ]
]:
    """Establish a gRPC connection to a gRPC server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on the same
        machine on port 8080, then `server_address` would be `"0.0.0.0:8080"` or
        `"[::]:8080"`.
    max_message_length : int
        The maximum length of gRPC messages that can be exchanged with the Flower
        server. The default should be sufficient for most models. Users who train
        very large models might need to increase this value. Note that the Flower
        server needs to be started with the same value
        (see `flwr.server.start_server`), otherwise it will not know about the
        increased limit and block larger messages.
        (default: 536_870_912, this equals 512MB)
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.

    Returns
    -------
    receive, send : Callable, Callable

    Examples
    --------
    Establishing a SSL-enabled connection to the server:

    >>> from pathlib import Path
    >>> with grpc_connection(
    >>>     server_address,
    >>>     max_message_length=max_message_length,
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> ) as conn:
    >>>     receive, send = conn
    >>>     server_message = receive()
    >>>     # do something here
    >>>     send(client_message)
    """
    if isinstance(root_certificates, str):
        root_certificates = Path(root_certificates).read_bytes()

    channel = create_channel(
        server_address=server_address,
        root_certificates=root_certificates,
        max_message_length=max_message_length,
    )
    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)

    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    def receive() -> TaskIns:
        server_message = next(server_message_iterator)
        return TaskIns(
            task_id=str(uuid.uuid4()),
            group_id="",
            workload_id=0,
            task=Task(
                producer=Node(node_id=0, anonymous=True),
                consumer=Node(node_id=0, anonymous=True),
                ancestry=[],
                legacy_server_message=server_message,
            ),
        )

    def send(task_res: TaskRes) -> None:
        msg = task_res.task.legacy_client_message
        return queue.put(msg, block=False)

    try:
        # Yield methods
        yield (receive, send, None, None)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "gRPC channel closed")
