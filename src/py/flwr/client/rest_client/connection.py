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
"""Contextmanager managing a REST-based channel to the Flower server."""


from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple

import requests

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    GetTasksRequest,
    GetTasksResponse,
    TokenizedResult,
)
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


@contextmanager
def rest_not_a_connection(
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
    client_id: Optional[str] = None,
) -> Iterator[
    Tuple[Callable[[], Optional[ServerMessage]], Callable[[ClientMessage], None]]
]:
    """Primitives for request/response-based interaction with a server.

    One notable difference to the grpc_connection context manager is that
    `receive` can return `None`.

    Parameters
    ----------
    server_address : str
        The IPv6 address of the server. If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"[::]:8080"`.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[bytes] (default: None)
        Ignored, for now. TODO: enable secure connections

    Returns
    -------
    receive, send : Callable, Callable
    """

    base_url = f"http://{server_address}"

    ###########################################################################
    # receive/send functions
    ###########################################################################

    def receive() -> Optional[ServerMessage]:
        """Receive next task from server."""
        # Serialize ProtoBuf to bytes
        get_tasks_req_msg = GetTasksRequest()
        get_tasks_req_msg_bytes: bytes = get_tasks_req_msg.SerializeToString()

        # Request instructions (task) from server
        r = requests.post(
            f"{base_url}/api/1.1/tasks",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=get_tasks_req_msg_bytes,
        )
        print(f"[C-{client_id}] POST /api/1.1/tasks:", r.status_code, r.headers)
        if r.status_code != 200:
            return None

        # Deserialize ProtoBuf from bytes
        get_tasks_response_msg = GetTasksResponse()
        get_tasks_response_msg.ParseFromString(r.content)

        server_msg = ServerMessage()
        server_msg.CopyFrom(
            get_tasks_response_msg.tokenized_tasks.tokenized_tasks[
                0
            ].task.legacy_server_message
        )

        return server_msg

    def send(client_message: ClientMessage) -> None:
        """Send task result back to server."""

        # Serialize ProtoBuf to bytes
        results_req_msg = CreateResultsRequest()
        tokenized_result = TokenizedResult()

        tokenized_result.result.legacy_client_message.CopyFrom(client_message)
        results_req_msg.tokenized_results.append(tokenized_result)

        results_req_msg_bytes: bytes = results_req_msg.SerializeToString()

        # Send ClientMessage to server
        r = requests.post(
            f"{base_url}/api/1.1/results",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=results_req_msg_bytes,
        )
        print(f"[C-{client_id}] POST /api/1.1/results:", r.status_code, r.headers)

    # yield methods
    try:
        yield (receive, send)
    except Exception as e:
        print(e)
