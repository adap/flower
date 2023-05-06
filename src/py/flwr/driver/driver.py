# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""Flower driver service client."""


import sys
from logging import ERROR, INFO, WARN, WARNING
from typing import Optional, Tuple

import grpc

from flwr.common import EventType, event
from flwr.common.address import parse_address
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto import driver_pb2, driver_pb2_grpc
from flwr.server.app import ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import FedAvg, Strategy

from .driver_client_manager import DriverClientManager

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `Driver` instance before calling any of the other `Driver`
methods.
"""


def start_driver(  # pylint: disable=too-many-arguments
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[DriverClientManager] = None,
    certificates: Optional[bytes] = None,
) -> History:
    """Start a Flower Driver API server.

    Parameters
    ----------
    server_address : Optional[str]
        The IPv4 or IPv6 address of the Driver API server.
        Defaults to `"[::]:8080"`.
    server : Optional[flwr.server.Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, then `start_driver` will create
        one.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None).
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `start_server` will use `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.driver.DriverClientManager] (default: None)
        An implementation of the class
        `flwr.server.DriverClientManager`. If no implementation is provided, then
        `start_driver` will use
        `flwr.server.client_manager.DriverClientManager(anonymous=False)`.
    certificates : bytes (default: None)
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

    >>> start_driver()

    Starting an SSL-enabled server:

    >>> start_driver(
    >>>     certificates=Path("/crts/root.pem").read_bytes()
    >>> )
    """
    event(EventType.START_SERVER_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Create the Driver and DriverClientManager if None is provided
    if client_manager is None:
        driver = Driver(driver_service_address=address, certificates=certificates)
        driver.connect()

        client_manager = DriverClientManager(driver=driver, anonymous=False)

    # Initialize the Driver API server and config
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

    # Start training
    hist = _fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the Driver API server
    client_manager.driver.disconnect()

    event(EventType.START_SERVER_LEAVE)

    return hist


def _init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: ClientManager,
) -> Tuple[Server, ServerConfig]:
    # Create server instance if none was given
    if server is None:
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
    log(INFO, "app_fit: metrics_distributed_fit %s", str(hist.metrics_distributed_fit))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist


class Driver:
    """`Driver` provides access to the Driver API."""

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        certificates: Optional[bytes] = None,
    ) -> None:
        self.driver_service_address = driver_service_address
        self.certificates = certificates
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[driver_pb2_grpc.DriverStub] = None

    def connect(self) -> None:
        """Connect to the Driver API."""
        event(EventType.DRIVER_CONNECT)
        if self.channel is not None or self.stub is not None:
            log(WARNING, "Already connected")
            return
        self.channel = create_channel(
            server_address=self.driver_service_address,
            root_certificates=self.certificates,
        )
        self.stub = driver_pb2_grpc.DriverStub(self.channel)
        log(INFO, "[Driver] Connected to %s", self.driver_service_address)

    def disconnect(self) -> None:
        """Disconnect from the Driver API."""
        event(EventType.DRIVER_DISCONNECT)
        if self.channel is None or self.stub is None:
            log(WARNING, "Already disconnected")
            return
        channel = self.channel
        self.channel = None
        self.stub = None
        channel.close()
        log(INFO, "[Driver] Disconnected")

    def get_nodes(self, req: driver_pb2.GetNodesRequest) -> driver_pb2.GetNodesResponse:
        """Get client IDs."""

        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Call Driver API
        res: driver_pb2.GetNodesResponse = self.stub.GetNodes(request=req)
        return res

    def push_task_ins(
        self, req: driver_pb2.PushTaskInsRequest
    ) -> driver_pb2.PushTaskInsResponse:
        """Schedule tasks."""

        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Call Driver API
        res: driver_pb2.PushTaskInsResponse = self.stub.PushTaskIns(request=req)
        return res

    def pull_task_res(
        self, req: driver_pb2.PullTaskResRequest
    ) -> driver_pb2.PullTaskResResponse:
        """Get task results."""

        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Call Driver API
        res: driver_pb2.PullTaskResResponse = self.stub.PullTaskRes(request=req)
        return res
