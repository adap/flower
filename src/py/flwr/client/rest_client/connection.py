# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
from typing import Callable, Iterator, Optional, Tuple, Type, TypeVar, Union

from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.message import Message as GrpcMessage

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
from flwr.common.serde import (
    message_from_taskins,
    message_to_taskres,
    record_value_dict_from_proto,
)
from flwr.common.typing import Run
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
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
PATH_GET_RUN: str = "/api/v0/fleet/get-run"

T = TypeVar("T", bound=GrpcMessage)


@contextmanager
def http_request_response(  # pylint: disable=,R0913, R0914, R0915
    server_address: str,
    insecure: bool,  # pylint: disable=unused-argument
    retry_invoker: RetryInvoker,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: Optional[
        Union[bytes, str]
    ] = None,  # pylint: disable=unused-argument
    authentication_keys: Optional[  # pylint: disable=unused-argument
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
) -> Iterator[
    Tuple[
        Callable[[], Optional[Message]],
        Callable[[Message], None],
        Optional[Callable[[], None]],
        Optional[Callable[[], None]],
        Optional[Callable[[int], Run]],
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
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Client authentication is not supported for this transport type.

    Returns
    -------
    receive : Callable
    send : Callable
    create_node : Optional[Callable]
    delete_node : Optional[Callable]
    get_run : Optional[Callable]
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
    if authentication_keys is not None:
        log(ERROR, "Client authentication is not supported for this transport type.")

    # Shared variables for inner functions
    metadata: Optional[Metadata] = None
    node: Optional[Node] = None
    ping_thread: Optional[threading.Thread] = None
    ping_stop_event = threading.Event()

    ###########################################################################
    # ping/create_node/delete_node/receive/send/get_run functions
    ###########################################################################

    def _request(
        req: GrpcMessage, res_type: Type[T], api_path: str, retry: bool = True
    ) -> Optional[T]:
        # Serialize the request
        req_bytes = req.SerializeToString()

        # Send the request
        def post() -> requests.Response:
            return requests.post(
                f"{base_url}/{api_path}",
                data=req_bytes,
                headers={
                    "Accept": "application/protobuf",
                    "Content-Type": "application/protobuf",
                },
                verify=verify,
                timeout=None,
            )

        if retry:
            res: requests.Response = retry_invoker.invoke(post)
        else:
            res = post()

        # Check status code and headers
        if res.status_code != 200:
            return None
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                api_path,
            )
            return None
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                api_path,
            )
            return None

        # Deserialize ProtoBuf from bytes
        grpc_res = res_type()
        grpc_res.ParseFromString(res.content)
        return grpc_res

    def ping() -> None:
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Construct the ping request
        req = PingRequest(node=node, ping_interval=PING_DEFAULT_INTERVAL)

        # Send the request
        res = _request(req, PingResponse, PATH_PING, retry=False)
        if res is None:
            return

        # Check if success
        if not res.success:
            raise RuntimeError("Ping failed unexpectedly.")

        # Wait
        rd = random.uniform(*PING_RANDOM_RANGE)
        next_interval: float = PING_DEFAULT_INTERVAL - PING_CALL_TIMEOUT
        next_interval *= PING_BASE_MULTIPLIER + rd
        if not ping_stop_event.is_set():
            ping_stop_event.wait(next_interval)

    def create_node() -> None:
        """Set create_node."""
        req = CreateNodeRequest(ping_interval=PING_DEFAULT_INTERVAL)

        # Send the request
        res = _request(req, CreateNodeResponse, PATH_CREATE_NODE)
        if res is None:
            return

        # Remember the node and the ping-loop thread
        nonlocal node, ping_thread
        node = res.node
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
        req = DeleteNodeRequest(node=node)

        # Send the request
        res = _request(req, DeleteNodeResponse, PATH_CREATE_NODE)
        if res is None:
            return

        # Cleanup
        node = None

    def receive() -> Optional[Message]:
        """Receive next task from server."""
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return None

        # Request instructions (task) from server
        req = PullTaskInsRequest(node=node)

        # Send the request
        res = _request(req, PullTaskInsResponse, PATH_PULL_TASK_INS)
        if res is None:
            return None

        # Get the current TaskIns
        task_ins: Optional[TaskIns] = get_task_ins(res)

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
        metadata = None

        # Construct TaskRes
        task_res = message_to_taskres(message)

        # Serialize ProtoBuf to bytes
        req = PushTaskResRequest(task_res_list=[task_res])

        # Send the request
        res = _request(req, PushTaskResResponse, PATH_PUSH_TASK_RES)
        if res is None:
            return

        log(
            INFO,
            "[Node] POST /%s: success, created result %s",
            PATH_PUSH_TASK_RES,
            res.results,  # pylint: disable=no-member
        )

    def get_run(run_id: int) -> Run:
        # Construct the request
        req = GetRunRequest(run_id=run_id)

        # Send the request
        res = _request(req, GetRunResponse, PATH_GET_RUN)
        if res is None:
            return Run(run_id, "", "", {})

        return Run(
            run_id,
            res.run.fab_id,
            res.run.fab_version,
            record_value_dict_from_proto(res.run.override_config),
        )

    try:
        # Yield methods
        yield (receive, send, create_node, delete_node, get_run)
    except Exception as exc:  # pylint: disable=broad-except
        log(ERROR, exc)
