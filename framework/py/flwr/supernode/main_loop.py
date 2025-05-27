# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Main loop for Flower SuperNode."""


import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging import INFO, WARN
from os import urandom
from pathlib import Path
import secrets
from typing import Callable, Optional, Union
from flwr.common.config import get_fused_config_from_fab
from flwr.common.exit_handlers import register_exit_handlers
import grpc
from .scheduler import run_scheduler
from cryptography.hazmat.primitives.asymmetric import ec
from grpc import RpcError

from flwr.app.error import Error
from flwr.cli.config_utils import get_fab_metadata
from flwr.client.clientapp.app import flwr_clientapp
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.supercore.object_store import ObjectStoreFactory, ObjectStore
from flwr.supernode.clientappio_servicer import ClientAppIoServicer
from flwr.client.grpc_adapter_client.connection import grpc_adapter
from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Message, Context, RecordDict, EventType
from flwr.common.address import parse_address
from flwr.common.constant import (
    CLIENT_OCTET,
    CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
    ISOLATION_MODE_SUBPROCESS,
    MAX_RETRY_DELAY,
    RUN_ID_NUM_BYTES,
    SERVER_OCTET,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPES,
    ErrorCode,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential
from flwr.common.typing import Fab, Run, RunNotRunningException, UserConfig
from flwr.proto.clientappio_pb2_grpc import add_ClientAppIoServicer_to_server
from flwr.supernode.nodestate import NodeStateFactory, NodeState


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def main_loop(
    *,
    superlink_fleet_api_address: str,
    node_config: UserConfig,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: str,
    authentication_keys: Optional[
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
    flwr_path: Optional[Path] = None,
    isolation: str = ISOLATION_MODE_SUBPROCESS,
    clientappio_api_address: str = CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    node_config: UserConfig
        The configuration of the node.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : str
        Configure the transport layer. Allowed values:
        - 'grpc-rere': gRPC, request-response
        - 'grpc-adapter': gRPC via 3rd party adapter (experimental)
        - 'rest': HTTP (experimental)
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Tuple containing the elliptic curve private key and public key for
        authentication from the cryptography library.
        Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        Used to establish an authenticated connection with the server.
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.
    flwr_path: Optional[Path] (default: None)
        The fully resolved path containing installed Flower Apps.
    isolation : str (default: ISOLATION_MODE_SUBPROCESS)
        Isolation mode for `ClientApp`. Possible values are `subprocess` and
        `process`. If `subprocess`, the `ClientApp` runs in a subprocess started
        by the SueprNode and communicates using gRPC at the address
        `clientappio_api_address`. If `process`, the `ClientApp` runs in a separate
        isolated process and communicates using gRPC at the address
        `clientappio_api_address`.
    clientappio_api_address : str
        (default: `CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS`)
        The SuperNode gRPC server address.
    """
    if insecure is None:
        insecure = root_certificates is None

    # Initialize StateFactory
    state_factory = NodeStateFactory()

    # Initialize FfsFactory
    # TODO: Add support for custom FFS path
    ffs_factory = FfsFactory("tmp.ffs")

    # Initialize ObjectStoreFactory
    objectstore_factory = ObjectStoreFactory()

    # Start ClientAppIo API server
    clientappio_server = _run_clientappio_api_grpc(
        address=clientappio_api_address,
        state_factory=state_factory,
        ffs_factory=ffs_factory,
        objectstore_factory=objectstore_factory,
        certificates=None,
    )

    # Start ClientApp scheduler
    scheduler_process = None
    if isolation == ISOLATION_MODE_SUBPROCESS:
        log(INFO, "Starting ClientApp scheduler")
        scheduler_process = multiprocessing.get_context("spawn").Process(
            target=run_scheduler,
            kwargs={
                "supernode_clientappio_api_address": clientappio_api_address,
                "root_certificates": root_certificates,
                "insecure": insecure,
            },
        )
        scheduler_process.start()

    # Register handlers for graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERNODE_LEAVE,
        exit_message="SuperNode terminated gracefully.",
        grpc_servers=[clientappio_server],
    )

    try:
        with _init_connection(
            transport=transport,
            server_address=superlink_fleet_api_address,
            insecure=insecure,
            root_certificates=root_certificates,
            authentication_keys=authentication_keys,
            max_retries=max_retries,
            max_wait_time=max_wait_time,
        ) as conn:
            # Initialize connection and states
            receive, send, create_node, delete_node, get_run, get_fab = conn
            state = state_factory.state()
            ffs = ffs_factory.ffs()
            object_store = objectstore_factory.store()

            # Request node ID from SuperLink
            if (node_id := create_node()) is None:
                raise ValueError("Failed to register SuperNode with the SuperLink")
            state.set_node_id(node_id)

            # Start the main loop
            while True:
                # Pull and store a message from the SuperLink
                _pull_and_store_message(
                    state=state,
                    ffs=ffs,
                    object_store=object_store,
                    node_config=node_config,
                    receive=receive,
                    get_run=get_run,
                    get_fab=get_fab,
                )

                # Push reply messages to the SuperLink
                _push_messages(state=state, send=send)

                # Sleep for 3 seconds before the next iteration
                time.sleep(3)
    finally:
        if scheduler_process is not None:
            scheduler_process.terminate()



def _pull_and_store_message(
    state: NodeState,
    ffs: Ffs,
    object_store: ObjectStore,
    node_config: UserConfig,
    receive: Callable[[], Optional[Message]],
    get_run: Callable[[int], Run],
    get_fab: Callable[[str, int], Fab],
) -> None:
    """Pull a message from the SuperLink and store it in the state."""
    # Pull message
    if (message := receive()) is None:
        return

    # Log message reception
    log(INFO, "")
    if message.metadata.group_id:
        log(
            INFO,
            "[RUN %s, ROUND %s]",
            message.metadata.run_id,
            message.metadata.group_id,
        )
    else:
        log(INFO, "[RUN %s]", message.metadata.run_id)
    log(
        INFO,
        "Received: %s message %s",
        message.metadata.message_type,
        message.metadata.message_id,
    )

    # Ensure the run and FAB are available
    run_id = message.metadata.run_id
    try:
        # Check if the message is from an unknown run
        if (run_info := state.get_run(run_id)) is None:
            print("Run ID %s is unknown. Fetching run info from SuperLink." % run_id)
            # Pull run info from SuperLink
            print("Get run")
            run_info = get_run(run_id)
            state.store_run(run_info)

            # Pull and store the FAB
            print("Get FAB")
            fab = get_fab(run_info.fab_hash, run_id)
            ffs.put(fab.content, {})

            # Initialize the context
            print("Get FAB metadata")
            run_cfg = get_fused_config_from_fab(fab.content, run_info)
            run_ctx = Context(
                run_id=run_id,
                node_id=state.get_node_id(),
                node_config=node_config,
                state=RecordDict(),
                run_config=run_cfg,
            )
            print("Store context")
            state.store_context(run_ctx)
        print("Store message %s for run ID %s." % (message.metadata.message_id, run_id))
        # Store the message in the state
        # TODO: use real object ID instead of a random one
        object_id = secrets.token_hex(8)
        state.store_message(message, object_id)
        print("Message stored.")
    except RunNotRunningException:
        log(
            INFO,
            "Run ID %s is not in `RUNNING` status. "
            "Ignoring message %s.",
            run_id,
            message.metadata.message_id,
        )


def _push_messages(
    state: NodeState,
    send: Callable[[Message], None],
) -> None:
    """Push reply messages to the SuperLink."""
    # Get messages to send
    reply_messages = state.get_message(is_reply=True)
    
    for message in reply_messages.values():
        # Log message sending
        log(INFO, "")
        if message.metadata.group_id:
            log(
                INFO,
                "[RUN %s, ROUND %s]",
                message.metadata.run_id,
                message.metadata.group_id,
            )
        else:
            log(INFO, "[RUN %s]", message.metadata.run_id)
        log(
            INFO,
            "Sending: %s message",
            message.metadata.message_type,
        )
        print(message.metadata)

        # Send the message
        try:
            send(message)
            log(INFO, "Sent successfully")
        except RunNotRunningException:
            log(
                INFO,
                "Run ID %s is not in `RUNNING` status. "
                "Ignoring reply message %s.",
                message.metadata.run_id,
                message.metadata.message_id,
            )


@contextmanager
def _init_connection(  # pylint: disable=too-many-positional-arguments
    transport: str,
    server_address: str,
    insecure: bool,
    root_certificates: Optional[Union[bytes, str]] = None,
    authentication_keys: Optional[
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> Iterator[
    tuple[
        Callable[[], Optional[Message]],
        Callable[[Message], None],
        Callable[[], Optional[int]],
        Callable[[], None],
        Callable[[int], Run],
        Callable[[str, int], Fab],
    ]
]:
    """Establish a connection to the Fleet API server at SuperLink."""
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"SuperLink address ({server_address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Use either gRPC bidirectional streaming or REST request/response
    if transport == TRANSPORT_TYPE_REST:
        try:
            from requests.exceptions import ConnectionError as RequestsConnectionError

            from flwr.client.rest_client.connection import http_request_response
        except ModuleNotFoundError:
            flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            flwr_exit(ExitCode.SUPERNODE_REST_ADDRESS_INVALID)
        connection, error_type = http_request_response, RequestsConnectionError
    elif transport == TRANSPORT_TYPE_GRPC_RERE:
        connection, error_type = grpc_request_response, RpcError
    elif transport == TRANSPORT_TYPE_GRPC_ADAPTER:
        connection, error_type = grpc_adapter, RpcError
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES})"
        )

    # Create RetryInvoker
    retry_invoker = _make_fleet_connection_retry_invoker(
        max_retries=max_retries,
        max_wait_time=max_wait_time,
        connection_error_type=error_type,
    )

    # Establish connection
    with connection(
        address,
        insecure,
        retry_invoker,
        GRPC_MAX_MESSAGE_LENGTH,
        root_certificates,
        authentication_keys,
    ) as conn:
        yield conn


def _make_fleet_connection_retry_invoker(
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
    connection_error_type: type[Exception] = RpcError,
) -> RetryInvoker:
    """Create a retry invoker for fleet connection."""

    def _on_success(retry_state: RetryState) -> None:
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        if retry_state.tries == 1:
            log(WARN, "Connection attempt failed, retrying...")
        else:
            log(
                WARN,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )

    return RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=connection_error_type,
        max_tries=max_retries + 1 if max_retries is not None else None,
        max_time=max_wait_time,
        on_giveup=lambda retry_state: (
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_success=_on_success,
        on_backoff=_on_backoff,
    )


def _run_clientappio_api_grpc(
    address: str,
    state_factory: NodeStateFactory,
    ffs_factory: FfsFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run ClientAppIo API gRPC server."""
    clientappio_servicer: grpc.Server = ClientAppIoServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
        objectstore_factory=objectstore_factory,
    )
    clientappio_add_servicer_to_server_fn = add_ClientAppIoServicer_to_server
    clientappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            clientappio_servicer,
            clientappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )
    log(INFO, "Starting Flower ClientAppIo API server on %s", address)
    clientappio_grpc_server.start()
    return clientappio_grpc_server
