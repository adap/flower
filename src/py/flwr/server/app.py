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
"""Flower server app."""


import argparse
import sys
from dataclasses import dataclass
from logging import INFO, WARN
from signal import SIGINT, SIGTERM, signal
from types import FrameType
from typing import Optional, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.logger import log
from flwr.proto.driver_pb2_grpc import add_DriverServicer_to_server
from flwr.proto.transport_pb2_grpc import add_FlowerServiceServicer_to_server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.driver.driver_servicer import DriverServicer
from flwr.server.grpc_server.driver_client_manager import DriverClientManager
from flwr.server.grpc_server.flower_service_servicer import FlowerServiceServicer
from flwr.server.grpc_server.grpc_server import (
    generic_create_grpc_server,
    start_grpc_server,
)
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.state import InMemoryState, State
from flwr.server.strategy import FedAvg, Strategy

ADDRESS_DRIVER_API = "[::]:9091"
ADDRESS_FLEET_API_GRPC = "[::]:9092"


@dataclass
class ServerConfig:
    """Flower server config.

    All attributes have default values which allows users to configure
    just the ones they care about.
    """

    num_rounds: int = 1
    round_timeout: Optional[float] = None


def start_server(  # pylint: disable=too-many-arguments
    *,
    server_address: str = ADDRESS_FLEET_API_GRPC,
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

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
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
        server_address=server_address,
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
    hist = _fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist


def _init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Server, ServerConfig]:
    # Create server instance if none was given
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config


def _fl(
    server: Server,
    config: ServerConfig,
) -> History:
    # Fit model
    hist = server.fit(num_rounds=config.num_rounds, timeout=config.round_timeout)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist


def run_server() -> None:
    """Run Flower server."""

    args = _parse_args()

    log(INFO, "Starting Flower server")
    event(EventType.RUN_SERVER_ENTER)

    # Shared State
    state = InMemoryState()

    # Shared DriverClientManager
    driver_client_manager = DriverClientManager(
        state=state,
    )

    # Start Driver API
    driver_server = _run_driver_api_grpc(
        address=args.driver_api_address,
        state=state,
    )

    # Start Fleet API
    fleet_server = _run_fleet_api_grpc_bidi(
        address=args.fleet_api_address,
        driver_client_manager=driver_client_manager,
    )

    default_handlers = {
        SIGINT: None,
        SIGTERM: None,
    }

    def graceful_exit_handler(  # type: ignore
        signalnum,
        frame: FrameType,  # pylint: disable=unused-argument
    ) -> None:
        """Exit handler to be registered with signal.signal.

        When called will reset signal handler to original signal handler
        from default_handlers.
        """

        # Reset to default handler
        signal(signalnum, default_handlers[signalnum])

        event_res = event(EventType.RUN_SERVER_LEAVE)

        driver_server.stop(grace=1)
        fleet_server.stop(grace=1)

        # Ensure event has happend
        event_res.result()

        # Setup things for graceful exit
        sys.exit(0)

    default_handlers[SIGINT] = signal(SIGINT, graceful_exit_handler)  # type: ignore
    default_handlers[SIGTERM] = signal(SIGTERM, graceful_exit_handler)  # type: ignore

    driver_server.wait_for_termination()
    fleet_server.wait_for_termination()


def _run_driver_api_grpc(
    address: str,
    state: State,
) -> grpc.Server:
    """Run Driver API (gRPC, request-response)."""

    # Create Driver API gRPC server
    driver_servicer: grpc.Server = DriverServicer(
        state=state,
    )
    driver_add_servicer_to_server_fn = add_DriverServicer_to_server
    driver_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(driver_servicer, driver_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=None,
    )

    log(INFO, "Flower ECE: Starting Driver API (gRPC-rere) on %s", address)
    driver_grpc_server.start()

    return driver_grpc_server


def _run_fleet_api_grpc_bidi(
    address: str,
    driver_client_manager: DriverClientManager,
) -> grpc.Server:
    """Run Fleet API (gRPC, bidirectional streaming)."""

    # Create (legacy) Fleet API gRPC server
    fleet_servicer = FlowerServiceServicer(
        client_manager=driver_client_manager,
    )
    fleet_add_servicer_to_server_fn = add_FlowerServiceServicer_to_server
    fleet_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(fleet_servicer, fleet_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=None,
    )

    log(INFO, "Flower ECE: Starting Fleet API (gRPC-bidi) on %s", address)
    fleet_grpc_server.start()

    return fleet_grpc_server


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start a long-running Flower server")

    # Driver API
    parser.add_argument(
        "--driver-api-address",
        help=f"Driver API gRPC server address. Default: {ADDRESS_DRIVER_API}",
        default=ADDRESS_DRIVER_API,
    )

    # Fleet API
    parser.add_argument(
        "--fleet-api-address",
        help=f"Fleet API gRPC server address. Default: {ADDRESS_FLEET_API_GRPC}",
        default=ADDRESS_FLEET_API_GRPC,
    )

    return parser.parse_args()
