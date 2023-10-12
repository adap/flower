# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Flower driver app."""


import sys
import threading
import time
from logging import INFO
from typing import Dict, Optional

from flwr.common import EventType, event
from flwr.common.address import parse_address
from flwr.common.logger import log
from flwr.proto import driver_pb2
from flwr.server.app import ServerConfig, init_defaults, run_fl
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy

from .driver import Driver
from .driver_client_proxy import DriverClientProxy

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `Driver` instance before calling any of the other `Driver`
methods.
"""


def start_driver(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
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
        `flwr.driver.driver_client_manager.DriverClientManager`. If no
        implementation is provided, then `start_driver` will use
        `flwr.driver.driver_client_manager.DriverClientManager`.
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
    Starting a driver that connects to an insecure server:

    >>> start_driver()

    Starting a driver that connects to an SSL-enabled server:

    >>> start_driver(
    >>>     certificates=Path("/crts/root.pem").read_bytes()
    >>> )
    """
    event(EventType.START_DRIVER_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Create the Driver
    driver = Driver(driver_service_address=address, certificates=certificates)
    driver.connect()
    lock = threading.Lock()

    # Initialize the Driver API server and config
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

    # Start the thread updating nodes
    thread = threading.Thread(
        target=update_client_manager,
        args=(
            driver,
            initialized_server.client_manager(),
            lock,
        ),
    )
    thread.start()

    # Start training
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the Driver API server and the thread
    with lock:
        driver.disconnect()
    thread.join()

    event(EventType.START_SERVER_LEAVE)

    return hist


def update_client_manager(
    driver: Driver,
    client_manager: ClientManager,
    lock: threading.Lock,
) -> None:
    """Update the nodes list in the client manager.

    This function periodically communicates with the associated driver to get all
    node_ids. Each node_id is then converted into a `DriverClientProxy` instance
    and stored in the `registered_nodes` dictionary with node_id as key.

    New nodes will be added to the ClientManager via `client_manager.register()`,
    and dead nodes will be removed from the ClientManager via
    `client_manager.unregister()`.
    """
    # Request for workload_id
    workload_id = driver.create_workload(driver_pb2.CreateWorkloadRequest()).workload_id

    # Loop until the driver is disconnected
    registered_nodes: Dict[int, DriverClientProxy] = {}
    while True:
        with lock:
            # End the while loop if the driver is disconnected
            if driver.stub is None:
                break
            get_nodes_res = driver.get_nodes(
                req=driver_pb2.GetNodesRequest(workload_id=workload_id)
            )
        all_node_ids = {node.node_id for node in get_nodes_res.nodes}
        dead_nodes = set(registered_nodes).difference(all_node_ids)
        new_nodes = all_node_ids.difference(registered_nodes)

        # Unregister dead nodes
        for node_id in dead_nodes:
            client_proxy = registered_nodes[node_id]
            client_manager.unregister(client_proxy)
            del registered_nodes[node_id]

        # Register new nodes
        for node_id in new_nodes:
            client_proxy = DriverClientProxy(
                node_id=node_id,
                driver=driver,
                anonymous=False,
                workload_id=workload_id,
            )
            if client_manager.register(client_proxy):
                registered_nodes[node_id] = client_proxy
            else:
                raise RuntimeError("Could not register node.")

        # Sleep for 3 seconds
        time.sleep(3)
