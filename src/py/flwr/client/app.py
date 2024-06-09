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
"""Flower client app."""

import signal
import sys
import time
from dataclasses import dataclass
from logging import DEBUG, ERROR, INFO, WARN
from typing import Callable, ContextManager, Optional, Tuple, Type, Union

from cryptography.hazmat.primitives.asymmetric import ec
from grpc import RpcError

from flwr.client.client import Client
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.client.typing import ClientFn
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, Message, event
from flwr.common.address import parse_address
from flwr.common.constant import (
    MISSING_EXTRA_REST,
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPES,
    ErrorCode,
)
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.message import Error
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential

from .grpc_client.connection import grpc_connection
from .grpc_rere_client.connection import grpc_request_response
from .message_handler.message_handler import handle_control_message
from .node_state import NodeState
from .numpy_client import NumPyClient


def _check_actionable_client(
    client: Optional[Client], client_fn: Optional[ClientFn]
) -> None:
    if client_fn is None and client is None:
        raise ValueError(
            "Both `client_fn` and `client` are `None`, but one is required"
        )

    if client_fn is not None and client is not None:
        raise ValueError(
            "Both `client_fn` and `client` are provided, but only one is allowed"
        )


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def start_client(
    *,
    server_address: str,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    client_fn : Optional[ClientFn]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : bool (default: True)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.

    Examples
    --------
    Starting a gRPC client with an insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>> )

    Starting an SSL-enabled gRPC client using system certificates:

    >>> def client_fn(cid: str):
    >>>     return FlowerClient()
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     insecure=False,
    >>> )

    Starting an SSL-enabled gRPC client using provided certificates:

    >>> from pathlib import Path
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    event(EventType.START_CLIENT_ENTER)
    _start_client_internal(
        server_address=server_address,
        load_client_app_fn=None,
        client_fn=client_fn,
        client=client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
        authentication_keys=authentication_keys,
        max_retries=max_retries,
        max_wait_time=max_wait_time,
    )
    event(EventType.START_CLIENT_LEAVE)


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def _start_client_internal(
    *,
    server_address: str,
    load_client_app_fn: Optional[Callable[[], ClientApp]] = None,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    load_client_app_fn : Optional[Callable[[], ClientApp]] (default: None)
        A function that can be used to load a `ClientApp` instance.
    client_fn : Optional[ClientFn]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.
    """
    if insecure is None:
        insecure = root_certificates is None

    if load_client_app_fn is None:
        _check_actionable_client(client, client_fn)

        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                cid: str,  # pylint: disable=unused-argument
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise ValueError(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance

            client_fn = single_client_factory

        def _load_client_app() -> ClientApp:
            return ClientApp(client_fn=client_fn)

        load_client_app_fn = _load_client_app

    # At this point, only `load_client_app_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    # Initialize connection context manager
    connection, address, connection_error_type = _init_connection(
        transport, server_address
    )

    app_state_tracker = _AppStateTracker()

    def _on_sucess(retry_state: RetryState) -> None:
        run_tracker.is_connected = True
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        run_tracker.is_connected = False
        if retry_state.tries == 1:
            log(WARN, "Connection attempt failed, retrying...")
        else:
            log(
                DEBUG,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )

    retry_invoker = RetryInvoker(
        wait_gen_factory=exponential,
        recoverable_exceptions=connection_error_type,
        max_tries=max_retries,
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
        on_success=_on_sucess,
        on_backoff=_on_backoff,
    )

    node_state = NodeState()

    while not app_state_tracker.interrupt:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            retry_invoker,
            grpc_max_message_length,
            root_certificates,
            authentication_keys,
        ) as conn:
            # pylint: disable-next=W0612
            receive, send, create_node, delete_node, get_run = conn

            # Register node
            if create_node is not None:
                create_node()  # pylint: disable=not-callable

            app_state_tracker.register_signal_handler()
            while not app_state_tracker.interrupt:
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

                    # Register context for this run
                    node_state.register_context(run_id=message.metadata.run_id)

                    # Retrieve context for this run
                    context = node_state.retrieve_context(
                        run_id=message.metadata.run_id
                    )

                    # Create an error reply message that will never be used to prevent
                    # the used-before-assignment linting error
                    reply_message = message.create_error_reply(
                        error=Error(code=ErrorCode.UNKNOWN, reason="Unknown")
                    )

                    # Handle app loading and task message
                    try:
                        # Load ClientApp instance
                        client_app: ClientApp = load_client_app_fn()

                        # Execute ClientApp
                        reply_message = client_app(message=message, context=context)
                    except Exception as ex:  # pylint: disable=broad-exception-caught

                        # Legacy grpc-bidi
                        if transport in ["grpc-bidi", None]:
                            log(ERROR, "Client raised an exception.", exc_info=ex)
                            # Raise exception, crash process
                            raise ex

                        # Don't update/change NodeState

                        e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                        # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
                        reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                        exc_entity = "ClientApp"
                        if isinstance(ex, LoadClientAppError):
                            reason = (
                                "An exception was raised when attempting to load "
                                "`ClientApp`"
                            )
                            e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
                            exc_entity = "SuperNode"

                        if not app_state_tracker.interrupt:
                            log(
                                ERROR, "%s raised an exception", exc_entity, exc_info=ex
                            )

                        # Create error message
                        reply_message = message.create_error_reply(
                            error=Error(code=e_code, reason=reason)
                        )
                    else:
                        # No exception, update node state
                        node_state.update_context(
                            run_id=message.metadata.run_id,
                            context=context,
                        )

                    # Send
                    send(reply_message)
                    log(INFO, "Sent reply")

                except StopIteration:
                    sleep_duration = 0
                    break

            # Unregister node
            if delete_node is not None and run_tracker.is_connected:
                delete_node()  # pylint: disable=not-callable

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            del app_state_tracker
            break

        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


def start_numpy_client(
    *,
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.

    Warning
    -------
    This function is deprecated since 1.7.0. Use :code:`flwr.client.start_client`
    instead and first convert your :code:`NumPyClient` to type
    :code:`flwr.client.Client` by executing its :code:`to_client()` method.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on
        the same machine on port 8080, then `server_address` would be
        `"[::]:8080"`.
    client : flwr.client.NumPyClient
        An implementation of the abstract base class `flwr.client.NumPyClient`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : bytes (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)

    Examples
    --------
    Starting a gRPC client with an insecure server connection:

    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting an SSL-enabled gRPC client using system certificates:

    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     insecure=False,
    >>> )

    Starting an SSL-enabled gRPC client using provided certificates:

    >>> from pathlib import Path
    >>>
    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    mssg = (
        "flwr.client.start_numpy_client() is deprecated. \n\tInstead, use "
        "`flwr.client.start_client()` by ensuring you first call "
        "the `.to_client()` method as shown below: \n"
        "\tflwr.client.start_client(\n"
        "\t\tserver_address='<IP>:<PORT>',\n"
        "\t\tclient=FlowerClient().to_client(),"
        " # <-- where FlowerClient is of type flwr.client.NumPyClient object\n"
        "\t)\n"
        "\tUsing `start_numpy_client()` is deprecated."
    )

    warn_deprecated_feature(name=mssg)

    # Calling this function is deprecated. A warning is thrown.
    # We first need to convert the supplied client to `Client.`

    wrp_client = client.to_client()

    start_client(
        server_address=server_address,
        client=wrp_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
    )


def _init_connection(transport: Optional[str], server_address: str) -> Tuple[
    Callable[
        [
            str,
            bool,
            RetryInvoker,
            int,
            Union[bytes, str, None],
            Optional[Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]],
        ],
        ContextManager[
            Tuple[
                Callable[[], Optional[Message]],
                Callable[[Message], None],
                Optional[Callable[[], None]],
                Optional[Callable[[], None]],
                Optional[Callable[[int], Tuple[str, str]]],
            ]
        ],
    ],
    str,
    Type[Exception],
]:
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Set the default transport layer
    if transport is None:
        transport = TRANSPORT_TYPE_GRPC_BIDI

    # Use either gRPC bidirectional streaming or REST request/response
    if transport == TRANSPORT_TYPE_REST:
        try:
            from requests.exceptions import ConnectionError as RequestsConnectionError

            from .rest_client.connection import http_request_response
        except ModuleNotFoundError:
            sys.exit(MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            sys.exit(
                "When using the REST API, please provide `https://` or "
                "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
            )
        connection, error_type = http_request_response, RequestsConnectionError
    elif transport == TRANSPORT_TYPE_GRPC_RERE:
        connection, error_type = grpc_request_response, RpcError
    elif transport == TRANSPORT_TYPE_GRPC_BIDI:
        connection, error_type = grpc_connection, RpcError
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES})"
        )

    return connection, address, error_type


@dataclass
class _AppStateTracker:
    interrupt: bool = False
    is_connected: bool = False

    def register_signal_handler(self) -> None:
        """Register handlers for exit signals."""

        def signal_handler(sig, frame):  # type: ignore
            # pylint: disable=unused-argument
            self.interrupt = True
            raise StopIteration from None

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
