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
"""Utility functions for custom ClientManager via Driver APIs."""


import threading
import time
from typing import Dict

from flwr.proto import driver_pb2
from flwr.server.client_manager import ClientManager

from .driver import Driver
from .driver_client_proxy import DriverClientProxy


def client_manager_update(
    driver: Driver,
    workload_id: str,
    registered_nodes: Dict[int, DriverClientProxy],
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
    # Loop until the driver is disconnected
    while True:
        with lock:
            # End the while loop if the driver is disconnected.
            if driver.stub is None:
                break
            get_nodes_res = driver.get_nodes(
                req=driver_pb2.GetNodesRequest(workload_id=workload_id)
            )
        compare_and_update(
            get_nodes_res, driver, workload_id, registered_nodes, client_manager
        )
        # Sleep for 3 seconds
        time.sleep(3)


def compare_and_update(
    get_nodes_res: driver_pb2.GetNodesResponse,
    driver: Driver,
    workload_id: str,
    registered_nodes: Dict[int, DriverClientProxy],
    client_manager: ClientManager,
) -> None:
    """Compare node_ids in GetNodesResponse to registered_nodes and update.

    New nodes will be added to the ClientManager via `client_manager.register()`,
    and dead nodes will be removed from the ClientManager via
    `client_manager.unregister()`. The `registered_nodes` dictionary will be updated.
    """
    all_node_ids = set(get_nodes_res.node_ids)
    new_nodes = all_node_ids.difference(registered_nodes)

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

    # Unregister dead nodes
    dead_nodes = set(registered_nodes).difference(all_node_ids)
    for node_id in dead_nodes:
        client_proxy = registered_nodes[node_id]
        client_manager.unregister(client_proxy)
        del registered_nodes[node_id]
