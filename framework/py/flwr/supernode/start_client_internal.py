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
from typing import Callable, Optional, Union

import grpc
from cryptography.hazmat.primitives.asymmetric import ec
from grpc import RpcError

from flwr.app.error import Error
from flwr.cli.config_utils import get_fab_metadata
from flwr.client.clientapp.app import flwr_clientapp
from flwr.client.clientapp.clientappio_servicer import (
    ClientAppInputs,
    ClientAppIoServicer,
)
from flwr.client.grpc_adapter_client.connection import grpc_adapter
from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Message
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
from flwr.supernode.nodestate import NodeStateFactory


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def start_client_internal(
    *,
    server_address: str,
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

    _clientappio_grpc_server, clientappio_servicer = run_clientappio_api_grpc(
        address=clientappio_api_address,
        certificates=None,
    )

    # DeprecatedRunInfoStore gets initialized when the first connection is established
    run_info_store: Optional[DeprecatedRunInfoStore] = None
    state_factory = NodeStateFactory()
    state = state_factory.state()
    mp_spawn_context = multiprocessing.get_context("spawn")

    runs: dict[int, Run] = {}

    while True:
        sleep_duration: int = 0
        with _init_connection(
            transport=transport,
            server_address=server_address,
            insecure=insecure,
            root_certificates=root_certificates,
            authentication_keys=authentication_keys,
            max_retries=max_retries,
            max_wait_time=max_wait_time,
        ) as conn:
            receive, send, create_node, delete_node, get_run, get_fab = conn

            # Register node when connecting the first time
            if run_info_store is None:
                # Call create_node fn to register node
                # and store node_id in state
                if (node_id := create_node()) is None:
                    raise ValueError("Failed to register SuperNode with the SuperLink")
                state.set_node_id(node_id)
                run_info_store = DeprecatedRunInfoStore(
                    node_id=state.get_node_id(),
                    node_config=node_config,
                )

            # pylint: disable=too-many-nested-blocks
            while True:
                try:
                    # Receive
                    message = receive()
                    if message is None:
                        time.sleep(3)  # Wait for 3s before asking again
                        continue

                    log(INFO, "")
                    if len(message.metadata.group_id) > 0:
                        log(
                            INFO,
                            "[RUN %s, ROUND %s]",
                            message.metadata.run_id,
                            message.metadata.group_id,
                        )
                    log(
                        INFO,
                        "Received: %s message %s",
                        message.metadata.message_type,
                        message.metadata.message_id,
                    )

                    # Handle control message
                    out_message, sleep_duration = handle_control_message(message)
                    if out_message:
                        send(out_message)
                        break

                    # Get run info
                    run_id = message.metadata.run_id
                    if run_id not in runs:
                        runs[run_id] = get_run(run_id)

                    run: Run = runs[run_id]
                    if get_fab is not None and run.fab_hash:
                        fab = get_fab(run.fab_hash, run_id)
                        fab_id, fab_version = get_fab_metadata(fab.content)
                    else:
                        fab = None
                        fab_id, fab_version = run.fab_id, run.fab_version

                    run.fab_id, run.fab_version = fab_id, fab_version

                    # Register context for this run
                    run_info_store.register_context(
                        run_id=run_id,
                        run=run,
                        flwr_path=flwr_path,
                        fab=fab,
                    )

                    # Retrieve context for this run
                    context = run_info_store.retrieve_context(run_id=run_id)
                    # Create an error reply message that will never be used to prevent
                    # the used-before-assignment linting error
                    reply_message = Message(
                        Error(code=ErrorCode.UNKNOWN, reason="Unknown"),
                        reply_to=message,
                    )

                    # Two isolation modes:
                    # 1. `subprocess`: SuperNode is starting the ClientApp
                    #    process as a subprocess.
                    # 2. `process`: ClientApp process gets started separately
                    #    (via `flwr-clientapp`), for example, in a separate
                    #    Docker container.

                    # Generate SuperNode token
                    token = int.from_bytes(urandom(RUN_ID_NUM_BYTES), "little")

                    # Mode 1: SuperNode starts ClientApp as subprocess
                    start_subprocess = isolation == ISOLATION_MODE_SUBPROCESS

                    # Share Message and Context with servicer
                    clientappio_servicer.set_inputs(
                        clientapp_input=ClientAppInputs(
                            message=message,
                            context=context,
                            run=run,
                            fab=fab,
                            token=token,
                        ),
                        token_returned=start_subprocess,
                    )

                    if start_subprocess:
                        _octet, _colon, _port = clientappio_api_address.rpartition(":")
                        io_address = (
                            f"{CLIENT_OCTET}:{_port}"
                            if _octet == SERVER_OCTET
                            else clientappio_api_address
                        )
                        # Start ClientApp subprocess
                        command = [
                            "flwr-clientapp",
                            "--clientappio-api-address",
                            io_address,
                            "--token",
                            str(token),
                        ]
                        command.append("--insecure")

                        proc = mp_spawn_context.Process(
                            target=_run_flwr_clientapp,
                            args=(command, os.getpid()),
                            daemon=True,
                        )
                        proc.start()
                        proc.join()
                    else:
                        # Wait for output to become available
                        while not clientappio_servicer.has_outputs():
                            time.sleep(0.1)

                    outputs = clientappio_servicer.get_outputs()
                    reply_message, context = outputs.message, outputs.context

                    # Update node state
                    run_info_store.update_context(
                        run_id=run_id,
                        context=context,
                    )

                    # Send
                    send(reply_message)
                    log(INFO, "Sent reply")

                except RunNotRunningException:
                    log(INFO, "")
                    log(
                        INFO,
                        "SuperNode aborted sending the reply message. "
                        "Run ID %s is not in `RUNNING` status.",
                        run_id,
                    )
                    log(INFO, "")
            # pylint: enable=too-many-nested-blocks

            # Unregister node
            if delete_node is not None:
                delete_node()  # pylint: disable=not-callable

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break

        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


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


def _run_flwr_clientapp(args: list[str], main_pid: int) -> None:
    # Monitor the main process in case of SIGKILL
    def main_process_monitor() -> None:
        while True:
            time.sleep(1)
            if os.getppid() != main_pid:
                os.kill(os.getpid(), 9)

    threading.Thread(target=main_process_monitor, daemon=True).start()

    # Run the command
    sys.argv = args
    flwr_clientapp()


def run_clientappio_api_grpc(
    address: str,
    certificates: Optional[tuple[bytes, bytes, bytes]],
) -> tuple[grpc.Server, ClientAppIoServicer]:
    """Run ClientAppIo API gRPC server."""
    clientappio_servicer: grpc.Server = ClientAppIoServicer()
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
    log(INFO, "Starting Flower ClientAppIo gRPC server on %s", address)
    clientappio_grpc_server.start()
    return clientappio_grpc_server, clientappio_servicer
