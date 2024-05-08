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
import asyncio
import csv
import importlib.util
import sys
import threading
from logging import ERROR, INFO, WARN
from os.path import isfile
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import grpc
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common.constant import (
    MISSING_EXTRA_REST,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPE_VCE,
)
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    private_key_to_bytes,
    public_key_to_bytes,
    ssh_types_to_elliptic_curve,
)
from flwr.proto.fleet_pb2_grpc import (  # pylint: disable=E0611
    add_FleetServicer_to_server,
)

from .client_manager import ClientManager
from .history import History
from .server import Server, init_defaults, run_fl
from .server_config import ServerConfig
from .strategy import Strategy
from .superlink.driver.driver_grpc import run_driver_api_grpc
from .superlink.fleet.grpc_bidi.grpc_server import (
    generic_create_grpc_server,
    start_grpc_server,
)
from .superlink.fleet.grpc_rere.fleet_servicer import FleetServicer
from .superlink.fleet.grpc_rere.server_interceptor import AuthenticateServerInterceptor
from .superlink.fleet.vce import start_vce
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


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def run_superlink() -> None:
    """Run Flower SuperLink (Driver API and Fleet API)."""
    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

    args = _parse_args_run_superlink().parse_args()

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

    # Start Driver API
    driver_server: grpc.Server = run_driver_api_grpc(
        address=address,
        state_factory=state_factory,
        certificates=certificates,
    )

    grpc_servers = [driver_server]
    bckg_threads = []

    # Start Fleet API
    if args.fleet_api_type == TRANSPORT_TYPE_REST:
        if (
            importlib.util.find_spec("requests")
            and importlib.util.find_spec("starlette")
            and importlib.util.find_spec("uvicorn")
        ) is None:
            sys.exit(MISSING_EXTRA_REST)
        address_arg = args.rest_fleet_api_address
        parsed_address = parse_address(address_arg)
        if not parsed_address:
            sys.exit(f"Fleet IP address ({address_arg}) cannot be parsed.")
        host, port, _ = parsed_address
        fleet_thread = threading.Thread(
            target=_run_fleet_api_rest,
            args=(
                host,
                port,
                args.ssl_keyfile,
                args.ssl_certfile,
                state_factory,
                args.rest_fleet_api_workers,
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
            address=address,
            state_factory=state_factory,
            certificates=certificates,
            interceptors=interceptors,
        )
        grpc_servers.append(fleet_server)
    elif args.fleet_api_type == TRANSPORT_TYPE_VCE:
        f_stop = asyncio.Event()  # Does nothing
        _run_fleet_api_vce(
            num_supernodes=args.num_supernodes,
            client_app_attr=args.client_app,
            backend_name=args.backend,
            backend_config_json_stream=args.backend_config,
            app_dir=args.app_dir,
            state_factory=state_factory,
            f_stop=f_stop,
        )
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
    if not args.require_client_authentication:
        return None

    if certificates is None:
        sys.exit(
            "Client authentication only works over secure connections. "
            "Please provide certificate paths using '--certificates' when "
            "enabling '--require-client-authentication'."
        )

    client_keys_file_path = Path(args.require_client_authentication[0])
    if not client_keys_file_path.exists():
        sys.exit(
            "The provided path to the client public keys CSV file does not exist: "
            f"{client_keys_file_path}. "
            "Please provide the CSV file path containing known client public keys "
            "to '--require-client-authentication'."
        )

    client_public_keys: Set[bytes] = set()
    ssh_private_key = load_ssh_private_key(
        Path(args.require_client_authentication[1]).read_bytes(),
        None,
    )
    ssh_public_key = load_ssh_public_key(
        Path(args.require_client_authentication[2]).read_bytes()
    )

    try:
        server_private_key, server_public_key = ssh_types_to_elliptic_curve(
            ssh_private_key, ssh_public_key
        )
    except TypeError:
        sys.exit(
            "The file paths provided could not be read as a private and public "
            "key pair. Client authentication requires an elliptic curve public and "
            "private key pair. Please provide the file paths containing elliptic "
            "curve private and public keys to '--require-client-authentication'."
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
                        "Error: Unable to parse the public keys in the .csv "
                        "file. Please ensure that the .csv file contains valid "
                        "SSH public keys and try again."
                    )
        return (
            client_public_keys,
            server_private_key,
            server_public_key,
        )


def _try_obtain_certificates(
    args: argparse.Namespace,
) -> Optional[Tuple[bytes, bytes, bytes]]:
    # Obtain certificates
    if args.insecure:
        log(WARN, "Option `--insecure` was set. Starting insecure HTTP server.")
        certificates = None
    # Check if certificates are provided
    elif args.certificates:
        certificates = (
            Path(args.certificates[0]).read_bytes(),  # CA certificate
            Path(args.certificates[1]).read_bytes(),  # server certificate
            Path(args.certificates[2]).read_bytes(),  # server private key
        )
    else:
        sys.exit(
            "Certificates are required unless running in insecure mode. "
            "Please provide certificate paths with '--certificates' or run the server "
            "in insecure mode using '--insecure' if you understand the risks."
        )
    return certificates


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


# pylint: disable=too-many-arguments
def _run_fleet_api_vce(
    num_supernodes: int,
    client_app_attr: str,
    backend_name: str,
    backend_config_json_stream: str,
    app_dir: str,
    state_factory: StateFactory,
    f_stop: asyncio.Event,
) -> None:
    log(INFO, "Flower VCE: Starting Fleet API (VirtualClientEngine)")

    start_vce(
        num_supernodes=num_supernodes,
        client_app_attr=client_app_attr,
        backend_name=backend_name,
        backend_config_json_stream=backend_config_json_stream,
        state_factory=state_factory,
        app_dir=app_dir,
        f_stop=f_stop,
    )


# pylint: disable=import-outside-toplevel,too-many-arguments
def _run_fleet_api_rest(
    host: str,
    port: int,
    ssl_keyfile: Optional[str],
    ssl_certfile: Optional[str],
    state_factory: StateFactory,
    workers: int,
) -> None:
    """Run Driver API (REST-based)."""
    try:
        import uvicorn

        from flwr.server.superlink.fleet.rest_rere.rest_api import app as fast_api_app
    except ModuleNotFoundError:
        sys.exit(MISSING_EXTRA_REST)
    if workers != 1:
        raise ValueError(
            f"The supported number of workers for the Fleet API (REST server) is "
            f"1. Instead given {workers}. The functionality of >1 workers will be "
            f"added in the future releases."
        )
    log(INFO, "Starting Flower REST server")

    # See: https://www.starlette.io/applications/#accessing-the-app-instance
    fast_api_app.state.STATE_FACTORY = state_factory

    validation_exceptions = _validate_ssl_files(
        ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile
    )
    if any(validation_exceptions):
        # Starting with 3.11 we can use ExceptionGroup but for now
        # this seems to be the reasonable approach.
        raise ValueError(validation_exceptions)

    uvicorn.run(
        app="flwr.server.superlink.fleet.rest_rere.rest_api:app",
        port=port,
        host=host,
        reload=False,
        access_log=True,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        workers=workers,
    )


def _validate_ssl_files(
    ssl_keyfile: Optional[str], ssl_certfile: Optional[str]
) -> List[ValueError]:
    validation_exceptions = []

    if ssl_keyfile is not None and not isfile(ssl_keyfile):
        msg = "Path argument `--ssl-keyfile` does not point to a file."
        log(ERROR, msg)
        validation_exceptions.append(ValueError(msg))

    if ssl_certfile is not None and not isfile(ssl_certfile):
        msg = "Path argument `--ssl-certfile` does not point to a file."
        log(ERROR, msg)
        validation_exceptions.append(ValueError(msg))

    if not bool(ssl_keyfile) == bool(ssl_certfile):
        msg = (
            "When setting one of `--ssl-keyfile` and "
            "`--ssl-certfile`, both have to be used."
        )
        log(ERROR, msg)
        validation_exceptions.append(ValueError(msg))

    return validation_exceptions


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
        "--certificates",
        nargs=3,
        metavar=("CA_CERT", "SERVER_CERT", "PRIVATE_KEY"),
        type=str,
        help="Paths to the CA certificate, server certificate, and server private "
        "key, in that order. Note: The server can only be started without "
        "certificates by enabling the `--insecure` flag.",
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
        "--require-client-authentication",
        nargs=3,
        metavar=("CLIENT_KEYS", "SERVER_PRIVATE_KEY", "SERVER_PUBLIC_KEY"),
        type=str,
        help="Provide three file paths: (1) a .csv file containing a list of "
        "known client public keys for authentication, (2) the server's private "
        "key file, and (3) the server's public key file.",
    )


def _add_args_driver_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--driver-api-address",
        help="Driver API (gRPC) server address (IPv4, IPv6, or a domain name)",
        default=ADDRESS_DRIVER_API,
    )


def _add_args_fleet_api(parser: argparse.ArgumentParser) -> None:
    # Fleet API transport layer type
    ex_group = parser.add_mutually_exclusive_group()
    ex_group.add_argument(
        "--grpc-rere",
        action="store_const",
        dest="fleet_api_type",
        const=TRANSPORT_TYPE_GRPC_RERE,
        default=TRANSPORT_TYPE_GRPC_RERE,
        help="Start a Fleet API server (gRPC-rere)",
    )
    ex_group.add_argument(
        "--rest",
        action="store_const",
        dest="fleet_api_type",
        const=TRANSPORT_TYPE_REST,
        help="Start a Fleet API server (REST, experimental)",
    )

    ex_group.add_argument(
        "--vce",
        action="store_const",
        dest="fleet_api_type",
        const=TRANSPORT_TYPE_VCE,
        help="Start a Fleet API server (VirtualClientEngine)",
    )

    # Fleet API gRPC-rere options
    grpc_rere_group = parser.add_argument_group(
        "Fleet API (gRPC-rere) server options", ""
    )
    grpc_rere_group.add_argument(
        "--grpc-rere-fleet-api-address",
        help="Fleet API (gRPC-rere) server address (IPv4, IPv6, or a domain name)",
        default=ADDRESS_FLEET_API_GRPC_RERE,
    )

    # Fleet API REST options
    rest_group = parser.add_argument_group("Fleet API (REST) server options", "")
    rest_group.add_argument(
        "--rest-fleet-api-address",
        help="Fleet API (REST) server address (IPv4, IPv6, or a domain name)",
        default=ADDRESS_FLEET_API_REST,
    )
    rest_group.add_argument(
        "--ssl-certfile",
        help="Fleet API (REST) server SSL certificate file (as a path str), "
        "needed for using 'https'.",
        default=None,
    )
    rest_group.add_argument(
        "--ssl-keyfile",
        help="Fleet API (REST) server SSL private key file (as a path str), "
        "needed for using 'https'.",
        default=None,
    )
    rest_group.add_argument(
        "--rest-fleet-api-workers",
        help="Set the number of concurrent workers for the Fleet API REST server.",
        type=int,
        default=1,
    )

    # Fleet API VCE options
    vce_group = parser.add_argument_group("Fleet API (VCE) server options", "")
    vce_group.add_argument(
        "--client-app",
        help="For example: `client:app` or `project.package.module:wrapper.app`.",
    )
    vce_group.add_argument(
        "--num-supernodes",
        type=int,
        help="Number of simulated SuperNodes.",
    )
    vce_group.add_argument(
        "--backend",
        default="ray",
        type=str,
        help="Simulation backend that executes the ClientApp.",
    )
    vce_group.add_argument(
        "--backend-config",
        type=str,
        default='{"client_resources": {"num_cpus":1, "num_gpus":0.0}, "tensorflow": 0}',
        help='A JSON formatted stream, e.g \'{"<keyA>":<value>, "<keyB>":<value>}\' to '
        "configure a backend. Values supported in <value> are those included by "
        "`flwr.common.typing.ConfigsRecordValues`. ",
    )
    parser.add_argument(
        "--app-dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load"
        "ClientApp from there."
        " Default: current working directory.",
    )
