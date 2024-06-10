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
"""Flower server app."""

import argparse
import csv
import importlib.util
import sys
import threading
from logging import INFO, WARN
from os.path import isfile
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import grpc
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common.constant import (
    MISSING_EXTRA_REST,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
)
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.proto.fleet_pb2_grpc import (  # pylint: disable=E0611
    add_FleetServicer_to_server,
)
from flwr.proto.grpcadapter_pb2_grpc import add_GrpcAdapterServicer_to_server

from .client_manager import ClientManager
from .history import History
from .server import Server, init_defaults, run_fl
from .server_config import ServerConfig
from .strategy import Strategy
from .superlink.driver.driver_grpc import run_driver_api_grpc
from .superlink.fleet.grpc_adapter.grpc_adapter_servicer import GrpcAdapterServicer
from .superlink.fleet.grpc_bidi.grpc_server import (
    generic_create_grpc_server,
    start_grpc_server,
)
from .superlink.fleet.grpc_rere.fleet_servicer import FleetServicer
from .superlink.fleet.grpc_rere.server_interceptor import AuthenticateServerInterceptor
from .superlink.state import StateFactory

ADDRESS_DRIVER_API = "0.0.0.0:9091"
ADDRESS_FLEET_API_GRPC_RERE = "0.0.0.0:9092"
ADDRESS_FLEET_API_GRPC_BIDI = "[::]:8080"  # IPv6 to keep start_server compatible
ADDRESS_FLEET_API_REST = "0.0.0.0:9093"

DATABASE = ":flwr-in-memory-state:"


def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:
    """Start a Flower server using the gRPC transport layer.

    Parameters
    ----------
    server_address : Optional[str]
        The IPv4 or IPv6 address of the server. Defaults to `"[::]:8080"`.
    server : Optional[flwr.server.Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, then `start_server` will create
        one.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None).
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `start_server` will use `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class
        `flwr.server.ClientManager`. If no implementation is provided, then
        `start_server` will use
        `flwr.server.client_manager.SimpleClientManager`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower clients. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower clients need to be started with the
        same value (see `flwr.client.start_client`), otherwise clients will
        not know about the increased limit and block larger messages.
    certificates : Tuple[bytes, bytes, bytes] (default: None)
        Tuple containing root certificate, server certificate, and private key
        to start a secure SSL-enabled server. The tuple is expected to have
        three bytes elements in the following order:

            * CA certificate.
            * server certificate.
            * server private key.

    Returns
    -------
    hist : flwr.server.history.History
        Object containing training and evaluation metrics.

    Examples
    --------
    Starting an insecure server:

    >>> start_server()

    Starting an SSL-enabled server:

    >>> start_server(
    >>>     certificates=(
    >>>         Path("/crts/root.pem").read_bytes(),
    >>>         Path("/crts/localhost.crt").read_bytes(),
    >>>         Path("/crts/localhost.key").read_bytes()
    >>>     )
    >>> )
    """
    event(EventType.START_SERVER_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Initialize server and server config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist


def run_driver_api() -> None:
    """Run Flower server (Driver API)."""
    log(INFO, "Starting Flower server (Driver API)")
    # Running `flower-driver-api` is deprecated
    warn_deprecated_feature("flower-driver-api")
    log(WARN, "Use `flower-superlink` instead")
    event(EventType.RUN_DRIVER_API_ENTER)
    args = _parse_args_run_driver_api().parse_args()

    # Parse IP address
    parsed_address = parse_address(args.driver_api_address)
    if not parsed_address:
        sys.exit(f"Driver IP address ({args.driver_api_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Obtain certificates
    certificates = _try_obtain_certificates(args)

    # Initialize StateFactory
    state_factory = StateFactory(args.database)

    # Start server
    grpc_server: grpc.Server = run_driver_api_grpc(
        address=address,
        state_factory=state_factory,
        certificates=certificates,
    )

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_DRIVER_API_LEAVE,
        grpc_servers=[grpc_server],
        bckg_threads=[],
    )

    # Block
    grpc_server.wait_for_termination()


# pylint: disable=too-many-locals
def run_fleet_api() -> None:
    """Run Flower server (Fleet API)."""
    log(INFO, "Starting Flower server (Fleet API)")
    # Running `flower-fleet-api` is deprecated
    warn_deprecated_feature("flower-fleet-api")
    log(WARN, "Use `flower-superlink` instead")
    event(EventType.RUN_FLEET_API_ENTER)
    args = _parse_args_run_fleet_api().parse_args()

    # Obtain certificates
    certificates = _try_obtain_certificates(args)

    # Initialize StateFactory
    state_factory = StateFactory(args.database)

    grpc_servers = []
    bckg_threads = []

    address_arg = args.fleet_api_address
    parsed_address = parse_address(address_arg)
    if not parsed_address:
        sys.exit(f"Fleet IP address ({address_arg}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    num_workers = args.fleet_api_num_workers
    if num_workers != 1:
        log(
            WARN,
            "The Fleet API currently supports only 1 worker. "
            "You have specified %d workers. "
            "Support for multiple workers will be added in future releases. "
            "Proceeding with a single worker.",
            args.fleet_api_num_workers,
        )
        num_workers = 1

    # Start Fleet API
    if args.fleet_api_type == TRANSPORT_TYPE_REST:
        if (
            importlib.util.find_spec("requests")
            and importlib.util.find_spec("starlette")
            and importlib.util.find_spec("uvicorn")
        ) is None:
            sys.exit(MISSING_EXTRA_REST)

        _, ssl_certfile, ssl_keyfile = (
            certificates if certificates is not None else (None, None, None)
        )
        fleet_thread = threading.Thread(
            target=_run_fleet_api_rest,
            args=(
                host,
                port,
                ssl_keyfile,
                ssl_certfile,
                state_factory,
                num_workers,
            ),
        )
        fleet_thread.start()
        bckg_threads.append(fleet_thread)
    elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE:
        address_arg = args.grpc_rere_fleet_api_address
        parsed_address = parse_address(address_arg)
        if not parsed_address:
            sys.exit(f"Fleet IP address ({address_arg}) cannot be parsed.")
        host, port, is_v6 = parsed_address
        address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"
        fleet_server = _run_fleet_api_grpc_rere(
            address=address,
            state_factory=state_factory,
            certificates=certificates,
        )
        grpc_servers.append(fleet_server)
    else:
        raise ValueError(f"Unknown fleet_api_type: {args.fleet_api_type}")

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_FLEET_API_LEAVE,
        grpc_servers=grpc_servers,
        bckg_threads=bckg_threads,
    )

    # Block
    if len(grpc_servers) > 0:
        grpc_servers[0].wait_for_termination()
    elif len(bckg_threads) > 0:
        bckg_threads[0].join()


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def run_superlink() -> None:
    """Run Flower SuperLink (Driver API and Fleet API)."""
    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

    args = _parse_args_run_superlink().parse_args()

    # Parse IP address
    parsed_driver_address = parse_address(args.driver_api_address)
    if not parsed_driver_address:
        sys.exit(f"Driver IP address ({args.driver_api_address}) cannot be parsed.")
    driver_host, driver_port, driver_is_v6 = parsed_driver_address
    driver_address = (
        f"[{driver_host}]:{driver_port}"
        if driver_is_v6
        else f"{driver_host}:{driver_port}"
    )

    # Obtain certificates
    certificates = _try_obtain_certificates(args)

    # Initialize StateFactory
    state_factory = StateFactory(args.database)

    # Start Driver API
    driver_server: grpc.Server = run_driver_api_grpc(
        address=driver_address,
        state_factory=state_factory,
        certificates=certificates,
    )

    grpc_servers = [driver_server]
    bckg_threads = []
    if not args.fleet_api_address:
        args.fleet_api_address = (
            ADDRESS_FLEET_API_GRPC_RERE
            if args.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE
            else ADDRESS_FLEET_API_REST
        )
    parsed_fleet_address = parse_address(args.fleet_api_address)
    if not parsed_fleet_address:
        sys.exit(f"Fleet IP address ({args.fleet_api_address}) cannot be parsed.")
    fleet_host, fleet_port, fleet_is_v6 = parsed_fleet_address
    fleet_address = (
        f"[{fleet_host}]:{fleet_port}" if fleet_is_v6 else f"{fleet_host}:{fleet_port}"
    )

    num_workers = args.fleet_api_num_workers
    if num_workers != 1:
        log(
            WARN,
            "The Fleet API currently supports only 1 worker. "
            "You have specified %d workers. "
            "Support for multiple workers will be added in future releases. "
            "Proceeding with a single worker.",
            args.fleet_api_num_workers,
        )
        num_workers = 1

    # Start Fleet API
    if args.fleet_api_type == TRANSPORT_TYPE_REST:
        if (
            importlib.util.find_spec("requests")
            and importlib.util.find_spec("starlette")
            and importlib.util.find_spec("uvicorn")
        ) is None:
            sys.exit(MISSING_EXTRA_REST)

        _, ssl_certfile, ssl_keyfile = (
            certificates if certificates is not None else (None, None, None)
        )

        fleet_thread = threading.Thread(
            target=_run_fleet_api_rest,
            args=(
                fleet_host,
                fleet_port,
                ssl_keyfile,
                ssl_certfile,
                state_factory,
                num_workers,
            ),
        )
        fleet_thread.start()
        bckg_threads.append(fleet_thread)
    elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE:
        maybe_keys = _try_setup_client_authentication(args, certificates)
        interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None
        if maybe_keys is not None:
            (
                client_public_keys,
                server_private_key,
                server_public_key,
            ) = maybe_keys
            state = state_factory.state()
            state.store_client_public_keys(client_public_keys)
            state.store_server_private_public_key(
                private_key_to_bytes(server_private_key),
                public_key_to_bytes(server_public_key),
            )
            log(
                INFO,
                "Client authentication enabled with %d known public keys",
                len(client_public_keys),
            )
            interceptors = [AuthenticateServerInterceptor(state)]

        fleet_server = _run_fleet_api_grpc_rere(
            address=fleet_address,
            state_factory=state_factory,
            certificates=certificates,
            interceptors=interceptors,
        )
        grpc_servers.append(fleet_server)
    elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
        fleet_server = _run_fleet_api_grpc_adapter(
            address=fleet_address,
            state_factory=state_factory,
            certificates=certificates,
        )
        grpc_servers.append(fleet_server)
    else:
        raise ValueError(f"Unknown fleet_api_type: {args.fleet_api_type}")

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERLINK_LEAVE,
        grpc_servers=grpc_servers,
        bckg_threads=bckg_threads,
    )

    # Block
    while True:
        if bckg_threads:
            for thread in bckg_threads:
                if not thread.is_alive():
                    sys.exit(1)
        driver_server.wait_for_termination(timeout=1)


def _try_setup_client_authentication(
    args: argparse.Namespace,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
) -> Optional[Tuple[Set[bytes], ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]]:
    if (
        not args.auth_list_public_keys
        and not args.auth_superlink_private_key
        and not args.auth_superlink_public_key
    ):
        return None

    if (
        not args.auth_list_public_keys
        or not args.auth_superlink_private_key
        or not args.auth_superlink_public_key
    ):
        sys.exit(
            "Authentication requires providing file paths for "
            "'--auth-list-public-keys', '--auth-superlink-private-key' and "
            "'--auth-superlink-public-key'. Provide all three to enable authentication."
        )

    if certificates is None:
        sys.exit(
            "Authentication requires secure connections. "
            "Please provide certificate paths to `--ssl-certfile`, "
            "`--ssl-keyfile`, and `—-ssl-ca-certfile` and try again."
        )

    client_keys_file_path = Path(args.auth_list_public_keys)
    if not client_keys_file_path.exists():
        sys.exit(
            "The provided path to the known public keys CSV file does not exist: "
            f"{client_keys_file_path}. "
            "Please provide the CSV file path containing known public keys "
            "to '--auth-list-public-keys'."
        )

    client_public_keys: Set[bytes] = set()

    try:
        ssh_private_key = load_ssh_private_key(
            Path(args.auth_superlink_private_key).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the private key file in "
            "'--auth-superlink-private-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again."
        )

    try:
        ssh_public_key = load_ssh_public_key(
            Path(args.auth_superlink_public_key).read_bytes()
        )
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the public key file in "
            "'--auth-superlink-public-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again."
        )

    with open(client_keys_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for element in row:
                public_key = load_ssh_public_key(element.encode())
                if isinstance(public_key, ec.EllipticCurvePublicKey):
                    client_public_keys.add(public_key_to_bytes(public_key))
                else:
                    sys.exit(
                        "Error: Unable to parse the public keys in the CSV "
                        "file. Please ensure that the CSV file path points to a valid "
                        "known SSH public keys files and try again."
                    )
        return (
            client_public_keys,
            ssh_private_key,
            ssh_public_key,
        )


def _try_obtain_certificates(
    args: argparse.Namespace,
) -> Optional[Tuple[bytes, bytes, bytes]]:
    # Obtain certificates
    if args.insecure:
        log(WARN, "Option `--insecure` was set. Starting insecure HTTP server.")
        return None
    # Check if certificates are provided
    if args.fleet_api_type in [TRANSPORT_TYPE_GRPC_RERE, TRANSPORT_TYPE_GRPC_ADAPTER]:
        if args.ssl_certfile and args.ssl_keyfile and args.ssl_ca_certfile:
            if not isfile(args.ssl_ca_certfile):
                sys.exit("Path argument `--ssl-ca-certfile` does not point to a file.")
            if not isfile(args.ssl_certfile):
                sys.exit("Path argument `--ssl-certfile` does not point to a file.")
            if not isfile(args.ssl_keyfile):
                sys.exit("Path argument `--ssl-keyfile` does not point to a file.")
            certificates = (
                Path(args.ssl_ca_certfile).read_bytes(),  # CA certificate
                Path(args.ssl_certfile).read_bytes(),  # server certificate
                Path(args.ssl_keyfile).read_bytes(),  # server private key
            )
            return certificates
        if args.ssl_certfile or args.ssl_keyfile or args.ssl_ca_certfile:
            sys.exit(
                "You need to provide valid file paths to `--ssl-certfile`, "
                "`--ssl-keyfile`, and `—-ssl-ca-certfile` to create a secure "
                "connection in Fleet API server (gRPC-rere)."
            )
    if args.fleet_api_type == TRANSPORT_TYPE_REST:
        if args.ssl_certfile and args.ssl_keyfile:
            if not isfile(args.ssl_certfile):
                sys.exit("Path argument `--ssl-certfile` does not point to a file.")
            if not isfile(args.ssl_keyfile):
                sys.exit("Path argument `--ssl-keyfile` does not point to a file.")
            certificates = (
                b"",
                Path(args.ssl_certfile).read_bytes(),  # server certificate
                Path(args.ssl_keyfile).read_bytes(),  # server private key
            )
            return certificates
        if args.ssl_certfile or args.ssl_keyfile:
            sys.exit(
                "You need to provide valid file paths to `--ssl-certfile` "
                "and `--ssl-keyfile` to create a secure connection "
                "in Fleet API server (REST, experimental)."
            )
    sys.exit(
        "Certificates are required unless running in insecure mode. "
        "Please provide certificate paths to `--ssl-certfile`, "
        "`--ssl-keyfile`, and `—-ssl-ca-certfile` or run the server "
        "in insecure mode using '--insecure' if you understand the risks."
    )


def _run_fleet_api_grpc_rere(
    address: str,
    state_factory: StateFactory,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
    interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None,
) -> grpc.Server:
    """Run Fleet API (gRPC, request-response)."""
    # Create Fleet API gRPC server
    fleet_servicer = FleetServicer(
        state_factory=state_factory,
    )
    fleet_add_servicer_to_server_fn = add_FleetServicer_to_server
    fleet_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(fleet_servicer, fleet_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors,
    )

    log(INFO, "Flower ECE: Starting Fleet API (gRPC-rere) on %s", address)
    fleet_grpc_server.start()

    return fleet_grpc_server


def _run_fleet_api_grpc_adapter(
    address: str,
    state_factory: StateFactory,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run Fleet API (GrpcAdapter)."""
    # Create Fleet API gRPC server
    fleet_servicer = GrpcAdapterServicer(
        state_factory=state_factory,
    )
    fleet_add_servicer_to_server_fn = add_GrpcAdapterServicer_to_server
    fleet_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(fleet_servicer, fleet_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(INFO, "Flower ECE: Starting Fleet API (GrpcAdapter) on %s", address)
    fleet_grpc_server.start()

    return fleet_grpc_server


# pylint: disable=import-outside-toplevel,too-many-arguments
def _run_fleet_api_rest(
    host: str,
    port: int,
    ssl_keyfile: Optional[str],
    ssl_certfile: Optional[str],
    state_factory: StateFactory,
    num_workers: int,
) -> None:
    """Run Driver API (REST-based)."""
    try:
        import uvicorn

        from flwr.server.superlink.fleet.rest_rere.rest_api import app as fast_api_app
    except ModuleNotFoundError:
        sys.exit(MISSING_EXTRA_REST)

    log(INFO, "Starting Flower REST server")

    # See: https://www.starlette.io/applications/#accessing-the-app-instance
    fast_api_app.state.STATE_FACTORY = state_factory

    uvicorn.run(
        app="flwr.server.superlink.fleet.rest_rere.rest_api:app",
        port=port,
        host=host,
        reload=False,
        access_log=True,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        workers=num_workers,
    )


def _parse_args_run_driver_api() -> argparse.ArgumentParser:
    """Parse command line arguments for Driver API."""
    parser = argparse.ArgumentParser(
        description="Start a Flower Driver API server. "
        "This server will be responsible for "
        "receiving TaskIns from the Driver script and "
        "sending them to the Fleet API. Once the client nodes "
        "are done, they will send the TaskRes back to this Driver API server (through"
        " the Fleet API) which will then send them back to the Driver script.",
    )

    _add_args_common(parser=parser)
    _add_args_driver_api(parser=parser)

    return parser


def _parse_args_run_fleet_api() -> argparse.ArgumentParser:
    """Parse command line arguments for Fleet API."""
    parser = argparse.ArgumentParser(
        description="Start a Flower Fleet API server."
        "This server will be responsible for "
        "sending TaskIns (received from the Driver API) to the client nodes "
        "and of receiving TaskRes sent back from those same client nodes once "
        "they are done. Then, this Fleet API server can send those "
        "TaskRes back to the Driver API.",
    )

    _add_args_common(parser=parser)
    _add_args_fleet_api(parser=parser)

    return parser


def _parse_args_run_superlink() -> argparse.ArgumentParser:
    """Parse command line arguments for both Driver API and Fleet API."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperLink",
    )

    _add_args_common(parser=parser)
    _add_args_driver_api(parser=parser)
    _add_args_fleet_api(parser=parser)

    return parser


def _add_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server without HTTPS, regardless of whether certificate "
        "paths are provided. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Fleet API server SSL certificate file (as a path str) "
        "to create a secure connection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Fleet API server SSL private key file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--ssl-ca-certfile",
        help="Fleet API server SSL CA certificate file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--database",
        help="A string representing the path to the database "
        "file that will be opened. Note that passing ':memory:' "
        "will open a connection to a database that is in RAM, "
        "instead of on disk. If nothing is provided, "
        "Flower will just create a state in memory.",
        default=DATABASE,
    )
    parser.add_argument(
        "--auth-list-public-keys",
        type=str,
        help="A CSV file (as a path str) containing a list of known public "
        "keys to enable authentication.",
    )
    parser.add_argument(
        "--auth-superlink-private-key",
        type=str,
        help="The SuperLink's private key (as a path str) to enable authentication.",
    )
    parser.add_argument(
        "--auth-superlink-public-key",
        type=str,
        help="The SuperLink's public key (as a path str) to enable authentication.",
    )


def _add_args_driver_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--driver-api-address",
        help="Driver API (gRPC) server address (IPv4, IPv6, or a domain name).",
        default=ADDRESS_DRIVER_API,
    )


def _add_args_fleet_api(parser: argparse.ArgumentParser) -> None:
    # Fleet API transport layer type
    parser.add_argument(
        "--fleet-api-type",
        default=TRANSPORT_TYPE_GRPC_RERE,
        type=str,
        choices=[
            TRANSPORT_TYPE_GRPC_RERE,
            TRANSPORT_TYPE_REST,
            TRANSPORT_TYPE_GRPC_ADAPTER,
        ],
        help="Start a gRPC-rere or REST (experimental) Fleet API server.",
    )
    parser.add_argument(
        "--fleet-api-address",
        help="Fleet API server address (IPv4, IPv6, or a domain name).",
    )
    parser.add_argument(
        "--fleet-api-num-workers",
        default=1,
        type=int,
        help="Set the number of concurrent workers for the Fleet API server.",
    )
