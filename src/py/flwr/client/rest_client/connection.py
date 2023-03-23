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
from logging import ERROR, INFO, WARN
from typing import Callable, Dict, Iterator, Optional, Tuple

try:
    import requests
except ImportError as missing_dep:
    raise ImportError(
        "To use the REST API you must install the "
        "extra dependencies by running `pip install flwr['rest']`."
    ) from missing_dep

from flwr.client.message_handler.task_handler import get_server_message
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

PATH_PULL_TASK_INS: str = "api/v0/fleet/pull-task-ins"
PATH_PUSH_TASK_RES: str = "api/v0/fleet/push-task-res"


@contextmanager
def http_request_response(
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: Optional[bytes] = None,  # pylint: disable=unused-argument
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
        Ignored, for now.

    Returns
    -------
    receive, send : Callable, Callable
    """

    log(
        WARN,
        """
        EXPERIMENTAL: `rest` is an experimental feature, it might change
        considerably in future versions of Flower
        """,
    )

    base_url = f"http://{server_address}"

    # Necessary state to link TaskRes to TaskIns
    state: Dict[str, Optional[TaskIns]] = {"current_task_ins": None}

    ###########################################################################
    # receive/send functions
    ###########################################################################

    def receive() -> Optional[ServerMessage]:
        """Receive next task from server."""

        # Serialize ProtoBuf to bytes
        pull_task_ins_req_proto = PullTaskInsRequest(
            node=Node(node_id=0, anonymous=True),
        )
        pull_task_ins_req_bytes: bytes = pull_task_ins_req_proto.SerializeToString()

        # Request instructions (task) from server
        res = requests.post(
            url=f"{base_url}/{PATH_PULL_TASK_INS}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=pull_task_ins_req_bytes,
        )

        # Check status code and headers
        if res.status_code != 200:
            return None
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                PATH_PULL_TASK_INS,
            )
            return None
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                PATH_PULL_TASK_INS,
            )
            return None

        # Deserialize ProtoBuf from bytes
        pull_task_ins_response_proto = PullTaskInsResponse()
        pull_task_ins_response_proto.ParseFromString(res.content)

        # Remember the current TaskIns
        task_ins_server_message_tuple = get_server_message(pull_task_ins_response_proto)
        if task_ins_server_message_tuple is None:
            state["current_task_ins"] = None
            return None

        task_ins, server_message = task_ins_server_message_tuple

        # Remember `task_ins` until `task_res` is available
        state["current_task_ins"] = task_ins

        # Return the ServerMessage
        log(INFO, "[Node] POST /%s: success", PATH_PULL_TASK_INS)
        return server_message

    def send(client_message_proto: ClientMessage) -> None:
        """Send task result back to server."""

        if state["current_task_ins"] is None:
            log(ERROR, "No current TaskIns")
            return

        # Wrap ClientMessage in TaskRes
        task_res = TaskRes(
            task_id="",  # This will be generated by the server
            task=Task(
                producer=Node(node_id=0, anonymous=True),
                consumer=Node(node_id=0, anonymous=True),
                legacy_client_message=client_message_proto,
                ancestry=[state["current_task_ins"].task_id],
            ),
        )

        # Serialize ProtoBuf to bytes
        push_task_res_request_proto = PushTaskResRequest(task_res_list=[task_res])
        push_task_res_request_bytes: bytes = (
            push_task_res_request_proto.SerializeToString()
        )

        # Send ClientMessage to server
        res = requests.post(
            url=f"{base_url}/{PATH_PUSH_TASK_RES}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=push_task_res_request_bytes,
        )

        state["current_task_ins"] = None

        # Check status code and headers
        if res.status_code != 200:
            return
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                PATH_PUSH_TASK_RES,
            )
            return
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                PATH_PUSH_TASK_RES,
            )
            return

        # Deserialize ProtoBuf from bytes
        push_task_res_response_proto = PushTaskResResponse()
        push_task_res_response_proto.ParseFromString(res.content)
        log(
            INFO,
            "[Node] POST /%s: success, created result %s",
            PATH_PUSH_TASK_RES,
            push_task_res_response_proto.results,  # pylint: disable=no-member
        )

    # yield methods
    try:
        yield (receive, send)
    except Exception as exc:  # pylint: disable=broad-except
        log(ERROR, exc)
