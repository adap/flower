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
import multiprocessing
import multiprocessing.context
import os
import sys
import threading
from collections.abc import Sequence
from logging import DEBUG, INFO, WARN
from pathlib import Path
from time import sleep
from typing import Any, Optional

import grpc
import yaml
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_ssh_public_key

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common.args import try_obtain_server_certificates
from flwr.common.auth_plugin import ExecAuthPlugin
from flwr.common.config import get_flwr_dir, parse_config_args
from flwr.common.constant import (
    AUTH_TYPE_YAML_KEY,
    CLIENT_OCTET,
    EXEC_API_DEFAULT_SERVER_ADDRESS,
    FLEET_API_GRPC_BIDI_DEFAULT_ADDRESS,
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    FLEET_API_REST_DEFAULT_ADDRESS,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    SERVER_OCTET,
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
    SIMULATIONIO_API_DEFAULT_SERVER_ADDRESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    EventLogWriterType,
)
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    public_key_to_bytes,
)
from flwr.proto.fleet_pb2_grpc import (  # pylint: disable=E0611
    add_FleetServicer_to_server,
)
from flwr.proto.grpcadapter_pb2_grpc import add_GrpcAdapterServicer_to_server
from flwr.server.fleet_event_log_interceptor import FleetEventLogInterceptor
from flwr.server.serverapp.app import flwr_serverapp
from flwr.simulation.app import flwr_simulation
from flwr.superexec.app import load_executor
from flwr.superexec.exec_grpc import run_exec_api_grpc

from .client_manager import ClientManager
from .history import History
from .server import Server, init_defaults, run_fl
from .server_config import ServerConfig
from .strategy import Strategy
from .superlink.driver.serverappio_grpc import run_serverappio_api_grpc
from .superlink.ffs.ffs_factory import FfsFactory
from .superlink.fleet.grpc_adapter.grpc_adapter_servicer import GrpcAdapterServicer
from .superlink.fleet.grpc_bidi.grpc_server import start_grpc_server
from .superlink.fleet.grpc_rere.fleet_servicer import FleetServicer
from .superlink.fleet.grpc_rere.server_interceptor import AuthenticateServerInterceptor
from .superlink.linkstate import LinkStateFactory
from .superlink.simulation.simulationio_grpc import run_simulationio_api_grpc

DATABASE = ":flwr-in-memory-state:"
BASE_DIR = get_flwr_dir() / "superlink" / "ffs"


try:
    from flwr.ee import (
        add_ee_args_superlink,
        get_dashboard_server,
        get_exec_auth_plugins,
        get_exec_event_log_writer_plugins,
        get_fleet_event_log_writer_plugins,
    )
except ImportError:

    # pylint: disable-next=unused-argument
    def add_ee_args_superlink(parser: argparse.ArgumentParser) -> None:
        """Add EE-specific arguments to the parser."""

    def get_exec_auth_plugins() -> dict[str, type[ExecAuthPlugin]]:
        """Return all Exec API authentication plugins."""
        raise NotImplementedError("No authentication plugins are currently supported.")

    def get_exec_event_log_writer_plugins() -> dict[str, type[EventLogWriterPlugin]]:
        """Return all Exec API event log writer plugins."""
        raise NotImplementedError(
            "No event log writer plugins are currently supported."
        )

    def get_fleet_event_log_writer_plugins() -> dict[str, type[EventLogWriterPlugin]]:
        """Return all Fleet API event log writer plugins."""
        raise NotImplementedError(
            "No event log writer plugins are currently supported."
        )


def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    server_address: str = FLEET_API_GRPC_BIDI_DEFAULT_ADDRESS,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[tuple[bytes, bytes, bytes]] = None,
) -> History:
    """Start a Flower server using the gRPC transport layer.

    Warning
    -------
    This function is deprecated since 1.13.0. Use the :code:`flower-superlink` command
    instead to start a SuperLink.

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
    msg = (
        "flwr.server.start_server() is deprecated."
        "\n\tInstead, use the `flower-superlink` CLI command to start a SuperLink "
        "as shown below:"
        "\n\n\t\t$ flower-superlink --insecure"
        "\n\n\tTo view usage and all available options, run:"
        "\n\n\t\t$ flower-superlink --help"
        "\n\n\tUsing `start_server()` is deprecated."
    )
    warn_deprecated_feature(name=msg)

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

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.START_SERVER_LEAVE,
        exit_message="Flower server terminated gracefully.",
        grpc_servers=[grpc_server],
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
    """Run Flower SuperLink (ServerAppIo API and Fleet API)."""
    args = _parse_args_run_superlink().parse_args()

    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

    # Warn unused options
    if args.flwr_dir is not None:
        log(
            WARN, "The `--flwr-dir` option is currently not in use and will be ignored."
        )

    # Parse IP addresses
    serverappio_address, _, _ = _format_address(args.serverappio_api_address)
    exec_address, _, _ = _format_address(args.exec_api_address)
    simulationio_address, _, _ = _format_address(args.simulationio_api_address)

    # Obtain certificates
    certificates = try_obtain_server_certificates(args)

    # Disable the user auth TLS check if args.disable_oidc_tls_cert_verification is
    # provided
    verify_tls_cert = not getattr(args, "disable_oidc_tls_cert_verification", None)

    auth_plugin: Optional[ExecAuthPlugin] = None
    event_log_plugin: Optional[EventLogWriterPlugin] = None
    # Load the auth plugin if the args.user_auth_config is provided
    if cfg_path := getattr(args, "user_auth_config", None):
        auth_plugin = _try_obtain_exec_auth_plugin(Path(cfg_path), verify_tls_cert)
        # Enable event logging if the args.enable_event_log is True
        if args.enable_event_log:
            event_log_plugin = _try_obtain_exec_event_log_writer_plugin()

    # Initialize StateFactory
    state_factory = LinkStateFactory(args.database)

    # Initialize FfsFactory
    ffs_factory = FfsFactory(args.storage_dir)

    # Start Exec API
    executor = load_executor(args)
    exec_server: grpc.Server = run_exec_api_grpc(
        address=exec_address,
        state_factory=state_factory,
        ffs_factory=ffs_factory,
        executor=executor,
        certificates=certificates,
        config=parse_config_args(
            [args.executor_config] if args.executor_config else args.executor_config
        ),
        auth_plugin=auth_plugin,
        event_log_plugin=event_log_plugin,
    )
    grpc_servers = [exec_server]

    # Determine Exec plugin
    # If simulation is used, don't start ServerAppIo and Fleet APIs
    sim_exec = executor.__class__.__qualname__ == "SimulationEngine"
    bckg_threads: list[threading.Thread] = []

    if sim_exec:
        simulationio_server: grpc.Server = run_simulationio_api_grpc(
            address=simulationio_address,
            state_factory=state_factory,
            ffs_factory=ffs_factory,
            certificates=None,  # SimulationAppIo API doesn't support SSL yet
        )
        grpc_servers.append(simulationio_server)

    else:
        # Start ServerAppIo API
        serverappio_server: grpc.Server = run_serverappio_api_grpc(
            address=serverappio_address,
            state_factory=state_factory,
            ffs_factory=ffs_factory,
            certificates=None,  # ServerAppIo API doesn't support SSL yet
        )
        grpc_servers.append(serverappio_server)

        # Start Fleet API
        if not args.fleet_api_address:
            if args.fleet_api_type in [
                TRANSPORT_TYPE_GRPC_RERE,
                TRANSPORT_TYPE_GRPC_ADAPTER,
            ]:
                args.fleet_api_address = FLEET_API_GRPC_RERE_DEFAULT_ADDRESS
            elif args.fleet_api_type == TRANSPORT_TYPE_REST:
                args.fleet_api_address = FLEET_API_REST_DEFAULT_ADDRESS

        fleet_address, host, port = _format_address(args.fleet_api_address)

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

        if args.fleet_api_type == TRANSPORT_TYPE_REST:
            if (
                importlib.util.find_spec("requests")
                and importlib.util.find_spec("starlette")
                and importlib.util.find_spec("uvicorn")
            ) is None:
                flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)

            fleet_thread = threading.Thread(
                target=_run_fleet_api_rest,
                args=(
                    host,
                    port,
                    args.ssl_keyfile,
                    args.ssl_certfile,
                    state_factory,
                    ffs_factory,
                    num_workers,
                ),
                daemon=True,
            )
            fleet_thread.start()
            bckg_threads.append(fleet_thread)
        elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE:
            node_public_keys = _try_load_public_keys_node_authentication(args)
            auto_auth = True
            if node_public_keys is not None:
                auto_auth = False
                state = state_factory.state()
                state.clear_supernode_auth_keys()
                state.store_node_public_keys(node_public_keys)
                log(
                    INFO,
                    "Node authentication enabled with %d known public keys",
                    len(node_public_keys),
                )
            else:
                log(DEBUG, "Automatic node authentication enabled")

            interceptors = [AuthenticateServerInterceptor(state_factory, auto_auth)]
            if getattr(args, "enable_event_log", None):
                fleet_log_plugin = _try_obtain_fleet_event_log_writer_plugin()
                if fleet_log_plugin is not None:
                    interceptors.append(FleetEventLogInterceptor(fleet_log_plugin))
                    log(INFO, "Flower Fleet event logging enabled")

            fleet_server = _run_fleet_api_grpc_rere(
                address=fleet_address,
                state_factory=state_factory,
                ffs_factory=ffs_factory,
                certificates=certificates,
                interceptors=interceptors,
            )
            grpc_servers.append(fleet_server)
        elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
            fleet_server = _run_fleet_api_grpc_adapter(
                address=fleet_address,
                state_factory=state_factory,
                ffs_factory=ffs_factory,
                certificates=certificates,
            )
            grpc_servers.append(fleet_server)
        else:
            raise ValueError(f"Unknown fleet_api_type: {args.fleet_api_type}")

    if args.isolation == ISOLATION_MODE_SUBPROCESS:

        _octet, _colon, _port = serverappio_address.rpartition(":")
        io_address = (
            f"{CLIENT_OCTET}:{_port}" if _octet == SERVER_OCTET else serverappio_address
        )
        address_arg = (
            "--simulationio-api-address" if sim_exec else "--serverappio-api-address"
        )
        address = simulationio_address if sim_exec else io_address
        cmd = "flwr-simulation" if sim_exec else "flwr-serverapp"

        # Scheduler thread
        scheduler_th = threading.Thread(
            target=_flwr_scheduler,
            args=(
                state_factory,
                address_arg,
                address,
                cmd,
            ),
            daemon=True,
        )
        scheduler_th.start()
        bckg_threads.append(scheduler_th)

    # Add Dashboard server if available
    if dashboard_address := getattr(args, "dashboard_address", None):
        dashboard_address_str, _, _ = _format_address(dashboard_address)
        dashboard_server = get_dashboard_server(
            address=dashboard_address_str,
            state_factory=state_factory,
            certificates=None,
        )

        grpc_servers.append(dashboard_server)

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERLINK_LEAVE,
        exit_message="SuperLink terminated gracefully.",
        grpc_servers=grpc_servers,
    )

    # Block until a thread exits prematurely
    while all(thread.is_alive() for thread in bckg_threads):
        sleep(0.1)

    # Exit if any thread has exited prematurely
    # This code will not be reached if the SuperLink stops gracefully
    flwr_exit(ExitCode.SUPERLINK_THREAD_CRASH)


def _run_flwr_command(args: list[str], main_pid: int) -> None:
    # Monitor the main process in case of SIGKILL
    def main_process_monitor() -> None:
        while True:
            sleep(1)
            if os.getppid() != main_pid:
                os.kill(os.getpid(), 9)

    threading.Thread(target=main_process_monitor, daemon=True).start()

    # Run the command
    sys.argv = args
    if args[0] == "flwr-serverapp":
        flwr_serverapp()
    elif args[0] == "flwr-simulation":
        flwr_simulation()
    else:
        raise ValueError(f"Unknown command: {args[0]}")


def _flwr_scheduler(
    state_factory: LinkStateFactory,
    io_api_arg: str,
    io_api_address: str,
    cmd: str,
) -> None:
    log(DEBUG, "Started %s scheduler thread.", cmd)
    state = state_factory.state()
    run_id_to_proc: dict[int, multiprocessing.context.SpawnProcess] = {}

    # Use the "spawn" start method for multiprocessing.
    mp_spawn_context = multiprocessing.get_context("spawn")

    # Periodically check for a pending run in the LinkState
    while True:
        sleep(0.1)
        pending_run_id = state.get_pending_run_id()

        if pending_run_id and pending_run_id not in run_id_to_proc:

            log(
                INFO,
                "Launching %s subprocess. Connects to SuperLink on %s",
                cmd,
                io_api_address,
            )
            # Start subprocess
            command = [
                cmd,
                "--run-once",
                io_api_arg,
                io_api_address,
                "--insecure",
            ]

            proc = mp_spawn_context.Process(
                target=_run_flwr_command, args=(command, os.getpid()), daemon=True
            )
            proc.start()

            # Store the process
            run_id_to_proc[pending_run_id] = proc

        # Clean up finished processes
        for run_id, proc in list(run_id_to_proc.items()):
            if not proc.is_alive():
                del run_id_to_proc[run_id]


def _format_address(address: str) -> tuple[str, str, int]:
    parsed_address = parse_address(address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"Address ({address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    return (f"[{host}]:{port}" if is_v6 else f"{host}:{port}", host, port)


def _try_load_public_keys_node_authentication(
    args: argparse.Namespace,
) -> Optional[set[bytes]]:
    """Return a set of node public keys."""
    if args.auth_superlink_private_key or args.auth_superlink_public_key:
        log(
            WARN,
            "The `--auth-superlink-private-key` and `--auth-superlink-public-key` "
            "arguments are deprecated and will be removed in a future release. Node "
            "authentication no longer requires these arguments.",
        )

    if not args.auth_list_public_keys:
        return None

    node_keys_file_path = Path(args.auth_list_public_keys)
    if not node_keys_file_path.exists():
        sys.exit(
            "The provided path to the known public keys CSV file does not exist: "
            f"{node_keys_file_path}. "
            "Please provide the CSV file path containing known public keys "
            "to '--auth-list-public-keys'."
        )

    node_public_keys: set[bytes] = set()

    with open(node_keys_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for element in row:
                public_key = load_ssh_public_key(element.encode())
                if isinstance(public_key, ec.EllipticCurvePublicKey):
                    node_public_keys.add(public_key_to_bytes(public_key))
                else:
                    sys.exit(
                        "Error: Unable to parse the public keys in the CSV "
                        "file. Please ensure that the CSV file path points to a valid "
                        "known SSH public keys files and try again."
                    )
    return node_public_keys


def _try_obtain_exec_auth_plugin(
    config_path: Path, verify_tls_cert: bool
) -> Optional[ExecAuthPlugin]:
    # Load YAML file
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    # Load authentication configuration
    auth_config: dict[str, Any] = config.get("authentication", {})
    auth_type: str = auth_config.get(AUTH_TYPE_YAML_KEY, "")

    # Load authentication plugin
    try:
        all_plugins: dict[str, type[ExecAuthPlugin]] = get_exec_auth_plugins()
        auth_plugin_class = all_plugins[auth_type]
        return auth_plugin_class(
            user_auth_config_path=config_path, verify_tls_cert=verify_tls_cert
        )
    except KeyError:
        if auth_type != "":
            sys.exit(
                f'Authentication type "{auth_type}" is not supported. '
                "Please provide a valid authentication type in the configuration."
            )
        sys.exit("No authentication type is provided in the configuration.")
    except NotImplementedError:
        sys.exit("No authentication plugins are currently supported.")


def _try_obtain_exec_event_log_writer_plugin() -> Optional[EventLogWriterPlugin]:
    """Return an instance of the event log writer plugin."""
    try:
        all_plugins: dict[str, type[EventLogWriterPlugin]] = (
            get_exec_event_log_writer_plugins()
        )
        plugin_class = all_plugins[EventLogWriterType.STDOUT]
        return plugin_class()
    except KeyError:
        sys.exit("No event log writer plugin is provided.")
    except NotImplementedError:
        sys.exit("No event log writer plugins are currently supported.")


def _try_obtain_fleet_event_log_writer_plugin() -> Optional[EventLogWriterPlugin]:
    """Return an instance of the Fleet Servicer event log writer plugin."""
    try:
        all_plugins: dict[str, type[EventLogWriterPlugin]] = (
            get_fleet_event_log_writer_plugins()
        )
        plugin_class = all_plugins[EventLogWriterType.STDOUT]
        return plugin_class()
    except KeyError:
        sys.exit("No Fleet API event log writer plugin is provided.")
    except NotImplementedError:
        sys.exit("No Fleet API event log writer plugins are currently supported.")


def _run_fleet_api_grpc_rere(
    address: str,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
    interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None,
) -> grpc.Server:
    """Run Fleet API (gRPC, request-response)."""
    # Create Fleet API gRPC server
    fleet_servicer = FleetServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
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
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run Fleet API (GrpcAdapter)."""
    # Create Fleet API gRPC server
    fleet_servicer = GrpcAdapterServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
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
# pylint: disable=too-many-positional-arguments
def _run_fleet_api_rest(
    host: str,
    port: int,
    ssl_keyfile: Optional[str],
    ssl_certfile: Optional[str],
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    num_workers: int,
) -> None:
    """Run ServerAppIo API (REST-based)."""
    try:
        import uvicorn

        from flwr.server.superlink.fleet.rest_rere.rest_api import app as fast_api_app
    except ModuleNotFoundError:
        flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)

    log(INFO, "Starting Flower REST server")

    # See: https://www.starlette.io/applications/#accessing-the-app-instance
    fast_api_app.state.STATE_FACTORY = state_factory
    fast_api_app.state.FFS_FACTORY = ffs_factory

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


def _parse_args_run_superlink() -> argparse.ArgumentParser:
    """Parse command line arguments for both ServerAppIo API and Fleet API."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperLink",
    )

    _add_args_common(parser=parser)
    add_ee_args_superlink(parser=parser)
    _add_args_serverappio_api(parser=parser)
    _add_args_fleet_api(parser=parser)
    _add_args_exec_api(parser=parser)
    _add_args_simulationio_api(parser=parser)

    return parser


def _add_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server without HTTPS, regardless of whether certificate "
        "paths are provided. Data transmitted between the gRPC client and server "
        "is not encrypted. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--flwr-dir",
        default=None,
        help="""The path containing installed Flower Apps.
        The default directory is:

        - `$FLWR_HOME/` if `$FLWR_HOME` is defined
        - `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined
        - `$HOME/.flwr/` in all other cases
        """,
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
        "--isolation",
        default=ISOLATION_MODE_SUBPROCESS,
        required=False,
        choices=[
            ISOLATION_MODE_SUBPROCESS,
            ISOLATION_MODE_PROCESS,
        ],
        help="Isolation mode when running a `ServerApp` (`subprocess` by default, "
        "possible values: `subprocess`, `process`). Use `subprocess` to configure "
        "SuperLink to run a `ServerApp` in a subprocess. Use `process` to indicate "
        "that a separate independent process gets created outside of SuperLink.",
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
        "--storage-dir",
        help="The base directory to store the objects for the Flower File System.",
        default=BASE_DIR,
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
        help="This argument is deprecated and will be removed in a future release.",
    )
    parser.add_argument(
        "--auth-superlink-public-key",
        type=str,
        help="This argument is deprecated and will be removed in a future release.",
    )


def _add_args_serverappio_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--serverappio-api-address",
        default=SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
        help="ServerAppIo API (gRPC) server address (IPv4, IPv6, or a domain name). "
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS}.",
    )


def _add_args_fleet_api(parser: argparse.ArgumentParser) -> None:
    # Fleet API transport layer type
    parser.add_argument(
        "--fleet-api-type",
        default=TRANSPORT_TYPE_GRPC_RERE,
        type=str,
        choices=[
            TRANSPORT_TYPE_GRPC_RERE,
            TRANSPORT_TYPE_GRPC_ADAPTER,
            TRANSPORT_TYPE_REST,
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


def _add_args_exec_api(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for Exec API."""
    parser.add_argument(
        "--exec-api-address",
        help="Exec API server address (IPv4, IPv6, or a domain name) "
        f"By default, it is set to {EXEC_API_DEFAULT_SERVER_ADDRESS}.",
        default=EXEC_API_DEFAULT_SERVER_ADDRESS,
    )
    parser.add_argument(
        "--executor",
        help="For example: `deployment:exec` or `project.package.module:wrapper.exec`. "
        "The default is `flwr.superexec.deployment:executor`",
        default="flwr.superexec.deployment:executor",
    )
    parser.add_argument(
        "--executor-dir",
        help="The directory for the executor.",
        default=".",
    )
    parser.add_argument(
        "--executor-config",
        help="Key-value pairs for the executor config, separated by spaces. "
        "For example:\n\n`--executor-config 'verbose=true "
        'root-certificates="certificates/superlink-ca.crt"\'`',
    )


def _add_args_simulationio_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--simulationio-api-address",
        default=SIMULATIONIO_API_DEFAULT_SERVER_ADDRESS,
        help="SimulationIo API (gRPC) server address (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SIMULATIONIO_API_DEFAULT_SERVER_ADDRESS}.",
    )
