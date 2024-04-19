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
"""Contextmanager for a REST request-response channel to the Flower server."""


import random
import sys
import threading
from contextlib import contextmanager
from copy import copy
from logging import ERROR, INFO, WARN
from typing import Callable, Iterator, Optional, Tuple, Union

from flwr.client.heartbeat import start_ping_loop
from flwr.client.message_handler.message_handler import validate_out_message
from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import (
    MISSING_EXTRA_REST,
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import message_from_taskins, message_to_taskres
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

try:
    import requests
except ModuleNotFoundError:
    sys.exit(MISSING_EXTRA_REST)


PATH_CREATE_NODE: str = "api/v0/fleet/create-node"
PATH_DELETE_NODE: str = "api/v0/fleet/delete-node"
PATH_PULL_TASK_INS: str = "api/v0/fleet/pull-task-ins"
PATH_PUSH_TASK_RES: str = "api/v0/fleet/push-task-res"
PATH_PING: str = "api/v0/fleet/ping"


@contextmanager
def http_request_response(  # pylint: disable=R0914, R0915
    server_address: str,
    insecure: bool,  # pylint: disable=unused-argument
    retry_invoker: RetryInvoker,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: Optional[
        Union[bytes, str]
    ] = None,  # pylint: disable=unused-argument
) -> Iterator[
    Tuple[
        Callable[[], Optional[Message]],
        Callable[[Message], None],
        Optional[Callable[[], None]],
        Optional[Callable[[], None]],
    ]
]:
    """Primitives for request/response-based interaction with a server.

    One notable difference to the grpc_connection context manager is that
    `receive` can return `None`.

    Parameters
    ----------
    server_address : str
        The IPv6 address of the server with `http://` or `https://`.
        If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"http://[::]:8080"`.
    insecure : bool
        Unused argument present for compatibilty.
    retry_invoker: RetryInvoker
        `RetryInvoker` object that will try to reconnect the client to the server
        after REST connection errors. If None, the client will only try to
        reconnect once after a failure.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        Path of the root certificate. If provided, a secure
        connection using the certificates will be established to an SSL-enabled
        Flower server. Bytes won't work for the REST API.

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

    base_url = server_address

    # NEVER SET VERIFY TO FALSE
    # Otherwise any server can fake its identity
    # Please refer to:
    # https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
    verify: Union[bool, str] = True
    if isinstance(root_certificates, str):
        verify = root_certificates
    elif isinstance(root_certificates, bytes):
        log(
            ERROR,
            "For the REST API, the root certificates "
            "must be provided as a string path to the client.",
        )

    # Shared variables for inner functions
    metadata: Optional[Metadata] = None
    node: Optional[Node] = None
    ping_thread: Optional[threading.Thread] = None
    ping_stop_event = threading.Event()

    ###########################################################################
    # ping/create_node/delete_node/receive/send functions
    ###########################################################################

    def ping() -> None:
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Construct the ping request
        req = PingRequest(node=node, ping_interval=PING_DEFAULT_INTERVAL)
        req_bytes: bytes = req.SerializeToString()

        # Send the request
        res = requests.post(
            url=f"{base_url}/{PATH_PING}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=req_bytes,
            verify=verify,
            timeout=PING_CALL_TIMEOUT,
        )

        # Check status code and headers
        if res.status_code != 200:
            return
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                PATH_PING,
            )
            return
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                PATH_PING,
            )
            return

        # Deserialize ProtoBuf from bytes
        ping_res = PingResponse()
        ping_res.ParseFromString(res.content)

        # Check if success
        if not ping_res.success:
            raise RuntimeError("Ping failed unexpectedly.")

        # Wait
        rd = random.uniform(*PING_RANDOM_RANGE)
        next_interval: float = PING_DEFAULT_INTERVAL - PING_CALL_TIMEOUT
        next_interval *= PING_BASE_MULTIPLIER + rd
        if not ping_stop_event.is_set():
            ping_stop_event.wait(next_interval)

    def create_node() -> None:
        """Set create_node."""
        create_node_req_proto = CreateNodeRequest(ping_interval=PING_DEFAULT_INTERVAL)
        create_node_req_bytes: bytes = create_node_req_proto.SerializeToString()

        res = retry_invoker.invoke(
            requests.post,
            url=f"{base_url}/{PATH_CREATE_NODE}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=create_node_req_bytes,
            verify=verify,
            timeout=None,
        )

        # Check status code and headers
        if res.status_code != 200:
            return
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                PATH_CREATE_NODE,
            )
            return
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                PATH_CREATE_NODE,
            )
            return

        # Deserialize ProtoBuf from bytes
        create_node_response_proto = CreateNodeResponse()
        create_node_response_proto.ParseFromString(res.content)

        # Remember the node and the ping-loop thread
        nonlocal node, ping_thread
        node = create_node_response_proto.node
        ping_thread = start_ping_loop(ping, ping_stop_event)

    def delete_node() -> None:
        """Set delete_node."""
        nonlocal node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Stop the ping-loop thread
        ping_stop_event.set()
        if ping_thread is not None:
            ping_thread.join()

        # Send DeleteNode request
        delete_node_req_proto = DeleteNodeRequest(node=node)
        delete_node_req_req_bytes: bytes = delete_node_req_proto.SerializeToString()
        res = retry_invoker.invoke(
            requests.post,
            url=f"{base_url}/{PATH_DELETE_NODE}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=delete_node_req_req_bytes,
            verify=verify,
            timeout=None,
        )

        # Check status code and headers
        if res.status_code != 200:
            return
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                PATH_DELETE_NODE,
            )
            return
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                PATH_DELETE_NODE,
            )

        # Cleanup
        node = None

    def receive() -> Optional[Message]:
        """Receive next task from server."""
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return None

        # Request instructions (task) from server
        pull_task_ins_req_proto = PullTaskInsRequest(node=node)
        pull_task_ins_req_bytes: bytes = pull_task_ins_req_proto.SerializeToString()

        # Request instructions (task) from server
        res = retry_invoker.invoke(
            requests.post,
            url=f"{base_url}/{PATH_PULL_TASK_INS}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=pull_task_ins_req_bytes,
            verify=verify,
            timeout=None,
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

        # Get the current TaskIns
        task_ins: Optional[TaskIns] = get_task_ins(pull_task_ins_response_proto)

        # Discard the current TaskIns if not valid
        if task_ins is not None and not (
            task_ins.task.consumer.node_id == node.node_id
            and validate_task_ins(task_ins)
        ):
            task_ins = None

        # Return the Message if available
        nonlocal metadata
        message = None
        if task_ins is not None:
            message = message_from_taskins(task_ins)
            metadata = copy(message.metadata)
            log(INFO, "[Node] POST /%s: success", PATH_PULL_TASK_INS)
        return message

    def send(message: Message) -> None:
        """Send task result back to server."""
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Get incoming message
        nonlocal metadata
        if metadata is None:
            log(ERROR, "No current message")
            return

        # Validate out message
        if not validate_out_message(message, metadata):
            log(ERROR, "Invalid out message")
            return

        # Construct TaskRes
        task_res = message_to_taskres(message)

        # Serialize ProtoBuf to bytes
        push_task_res_request_proto = PushTaskResRequest(task_res_list=[task_res])
        push_task_res_request_bytes: bytes = (
            push_task_res_request_proto.SerializeToString()
        )

        # Send ClientMessage to server
        res = retry_invoker.invoke(
            requests.post,
            url=f"{base_url}/{PATH_PUSH_TASK_RES}",
            headers={
                "Accept": "application/protobuf",
                "Content-Type": "application/protobuf",
            },
            data=push_task_res_request_bytes,
            verify=verify,
            timeout=None,
        )

        metadata = None

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

    try:
        # Yield methods
        yield (receive, send, create_node, delete_node)
    except Exception as exc:  # pylint: disable=broad-except
        log(ERROR, exc)
