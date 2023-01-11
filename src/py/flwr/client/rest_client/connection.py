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


import uuid
from contextlib import contextmanager
from logging import ERROR, INFO, WARN
from typing import Callable, Iterator, Optional, Tuple

import requests

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.fleet_pb2 import (
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage

PATH_PULL_TASK_INS: str = "/api/v0/fleet/pull-task-ins"
PATH_PUSH_TASK_RES: str = "/api/v0/fleet/push-task-res"


@contextmanager
def rest_not_a_connection(
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
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
    node_id : Optional[int]

    Returns
    -------
    receive, send : Callable, Callable
    """

    base_url = f"http://{server_address}"  # TODO handle HTTPS

    # Generate a random node_id that lives as long as the process lives
    node_id: int = uuid.uuid1().int >> 64

    # Necessary state to link TaskRes to TaskIns
    current_task_ins: Optional[TaskIns] = None

    ###########################################################################
    # receive/send functions
    ###########################################################################

    def receive() -> Optional[ServerMessage]:
        """Receive next task from server."""

        # Serialize ProtoBuf to bytes
        pull_task_ins_req_proto = PullTaskInsRequest(node_id=node_id)
        pull_task_ins_req_bytes: bytes = pull_task_ins_req_proto.SerializeToString()

        # Request instructions (task) from server
        r = requests.post(
            f"{base_url}/{PATH_PULL_TASK_INS}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=pull_task_ins_req_bytes,
        )
        log(
            INFO,
            "[Node %s] POST /%s: %s %s",
            node_id,
            PATH_PULL_TASK_INS,
            r.status_code,
            r.headers,
        )
        if r.status_code != 200:
            return None

        # Check headers
        if not "content-type" in r.headers:
            log(
                WARN,
                "[Node %s] POST /%s: missing header `Content-Type`",
                node_id,
                PATH_PULL_TASK_INS,
            )
            return None
        if r.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node %s] POST /%s: header `Content-Type` has wrong value",
                node_id,
                PATH_PULL_TASK_INS,
            )
            return None

        # Deserialize ProtoBuf from bytes
        pull_task_ins_response_proto = PullTaskInsResponse()
        pull_task_ins_response_proto.ParseFromString(r.content)

        # Extract a single ServerMessage from the response, if possible
        if len(pull_task_ins_response_proto.task_ins_set) == 0:
            log(
                INFO,
                "[Node %s] POST /%s: No TaskIns received",
                node_id,
                PATH_PULL_TASK_INS,
            )
            return None

        task_ins: TaskIns = pull_task_ins_response_proto.task_ins_set[
            0
        ]  # TODO handle multiple

        if task_ins.task.consumer.node_id != node_id:
            log(
                ERROR,
                "[Node %s] POST /%s: Task consumer node_id (%s) is different from local node_id (%s)",
                node_id,
                PATH_PULL_TASK_INS,
                task_ins.task.consumer.node_id,
                node_id,
            )
            return None

        if task_ins.task.legacy_server_message == None:
            log(
                ERROR,
                "[Node %s] POST /%s: legacy_server_message is None",
                node_id,
                PATH_PULL_TASK_INS,
            )
            return None

        # Remember the current TaskIns
        current_task_ins = task_ins
        server_message_proto: ServerMessage = task_ins.task.legacy_server_message

        # Return the ServerMessage
        log(INFO, "[Node {node_id}] POST /%s: success", PATH_PULL_TASK_INS)
        return server_message_proto

    def send(client_message_proto: ClientMessage) -> None:
        """Send task result back to server."""

        if current_task_ins is None:
            log(ERROR, "No current TaskIns")
            return

        # Wrap ClientMessage in TaskRes
        task_res = TaskRes(
            task_id="",  # This will be generated by the server
            task=Task(
                producer=Node(node_id=node_id, anonymous=False),
                legacy_client_message=client_message_proto,
                ancestry=[current_task_ins.task_id],
            ),
        )

        # Serialize ProtoBuf to bytes
        push_task_res_request_proto = PushTaskResRequest(
            node_id=node_id,
            task_res_set=[task_res],
        )
        push_task_res_request_bytes: bytes = (
            push_task_res_request_proto.SerializeToString()
        )

        # Send ClientMessage to server
        r = requests.post(
            f"{base_url}/{PATH_PUSH_TASK_RES}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=push_task_res_request_bytes,
        )
        log(
            INFO,
            "[Node %s] POST /%s:",
            PATH_PUSH_TASK_RES,
            node_id,
            r.status_code,
            r.headers,
        )

        # TODO check status code and response
        current_task_ins = None

    # yield methods
    try:
        yield (receive, send)
    except Exception as e:
        print(e)
