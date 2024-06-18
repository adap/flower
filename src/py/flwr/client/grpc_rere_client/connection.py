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
"""Contextmanager for a gRPC request-response channel to the Flower server."""


import random
import threading
from contextlib import contextmanager
from copy import copy
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple, Type, Union, cast

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.client.heartbeat import start_ping_loop
from flwr.client.message_handler.message_handler import validate_out_message
from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import (
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import message_from_taskins, message_to_taskres
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .client_interceptor import AuthenticateClientInterceptor
from .grpc_adapter import GrpcAdapter


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


@contextmanager
def grpc_request_response(  # pylint: disable=R0913, R0914, R0915
    server_address: str,
    insecure: bool,
    retry_invoker: RetryInvoker,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: Optional[Union[bytes, str]] = None,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    adapter_cls: Optional[Union[Type[FleetStub], Type[GrpcAdapter]]] = None,
) -> Iterator[
    Tuple[
        Callable[[], Optional[Message]],
        Callable[[Message], None],
        Optional[Callable[[], None]],
        Optional[Callable[[], None]],
        Optional[Callable[[int], Tuple[str, str]]],
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
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    retry_invoker: RetryInvoker
        `RetryInvoker` object that will try to reconnect the client to the server
        after gRPC errors. If None, the client will only try to
        reconnect once after a failure.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        Path of the root certificate. If provided, a secure
        connection using the certificates will be established to an SSL-enabled
        Flower server. Bytes won't work for the REST API.

    Returns
    -------
    receive : Callable
    send : Callable
    create_node : Optional[Callable]
    delete_node : Optional[Callable]
    get_run : Optional[Callable]
    """
    if isinstance(root_certificates, str):
        root_certificates = Path(root_certificates).read_bytes()

    interceptors: Optional[Sequence[grpc.UnaryUnaryClientInterceptor]] = None
    if authentication_keys is not None:
        interceptors = AuthenticateClientInterceptor(
            authentication_keys[0], authentication_keys[1]
        )

    channel = create_channel(
        server_address=server_address,
        insecure=insecure,
        root_certificates=root_certificates,
        max_message_length=max_message_length,
        interceptors=interceptors,
    )
    channel.subscribe(on_channel_state_change)

    # Shared variables for inner functions
    if adapter_cls is None:
        adapter_cls = FleetStub
    stub = adapter_cls(channel)
    metadata: Optional[Metadata] = None
    node: Optional[Node] = None
    ping_thread: Optional[threading.Thread] = None
    ping_stop_event = threading.Event()

    ###########################################################################
    # ping/create_node/delete_node/receive/send/get_run functions
    ###########################################################################

    def ping() -> None:
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Construct the ping request
        req = PingRequest(node=node, ping_interval=PING_DEFAULT_INTERVAL)

        # Call FleetAPI
        res: PingResponse = stub.Ping(req, timeout=PING_CALL_TIMEOUT)

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
        # Call FleetAPI
        create_node_request = CreateNodeRequest(ping_interval=PING_DEFAULT_INTERVAL)
        create_node_response = retry_invoker.invoke(
            stub.CreateNode,
            request=create_node_request,
        )

        # Remember the node and the ping-loop thread
        nonlocal node, ping_thread
        node = cast(Node, create_node_response.node)
        ping_thread = start_ping_loop(ping, ping_stop_event)

    def delete_node() -> None:
        """Set delete_node."""
        # Get Node
        nonlocal node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Stop the ping-loop thread
        ping_stop_event.set()

        # Call FleetAPI
        delete_node_request = DeleteNodeRequest(node=node)
        retry_invoker.invoke(stub.DeleteNode, request=delete_node_request)

        # Cleanup
        node = None

    def receive() -> Optional[Message]:
        """Receive next task from server."""
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return None

        # Request instructions (task) from server
        request = PullTaskInsRequest(node=node)
        response = retry_invoker.invoke(stub.PullTaskIns, request=request)

        # Get the current TaskIns
        task_ins: Optional[TaskIns] = get_task_ins(response)

        # Discard the current TaskIns if not valid
        if task_ins is not None and not (
            task_ins.task.consumer.node_id == node.node_id
            and validate_task_ins(task_ins)
        ):
            task_ins = None

        # Construct the Message
        in_message = message_from_taskins(task_ins) if task_ins else None

        # Remember `metadata` of the in message
        nonlocal metadata
        metadata = copy(in_message.metadata) if in_message else None

        # Return the message if available
        return in_message

    def send(message: Message) -> None:
        """Send task result back to server."""
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return

        # Get the metadata of the incoming message
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
        request = PushTaskResRequest(task_res_list=[task_res])
        _ = retry_invoker.invoke(stub.PushTaskRes, request)

        # Cleanup
        metadata = None

    def get_run(run_id: int) -> Tuple[str, str]:
        # Call FleetAPI
        get_run_request = GetRunRequest(run_id=run_id)
        get_run_response: GetRunResponse = retry_invoker.invoke(
            stub.GetRun,
            request=get_run_request,
        )

        # Return fab_id and fab_version
        return get_run_response.run.fab_id, get_run_response.run.fab_version

    try:
        # Yield methods
        yield (receive, send, create_node, delete_node, get_run)
    except Exception as exc:  # pylint: disable=broad-except
        log(ERROR, exc)
