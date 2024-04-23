# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for the `start_driver`."""


import threading
from typing import Dict, Tuple

from ..client_manager import ClientManager
from ..compat.grpc_driver_client_proxy import GrpcDriverClientProxy
from ..driver import Driver


def start_update_client_manager_thread(
    driver: Driver,
    client_manager: ClientManager,
) -> Tuple[threading.Thread, threading.Event]:
    """Periodically update the nodes list in the client manager in a thread.

    This function starts a thread that periodically uses the associated driver to
    get all node_ids. Each node_id is then converted into a `GrpcDriverClientProxy`
    instance and stored in the `registered_nodes` dictionary with node_id as key.

    New nodes will be added to the ClientManager via `client_manager.register()`,
    and dead nodes will be removed from the ClientManager via
    `client_manager.unregister()`.

    Parameters
    ----------
    driver : Driver
        The Driver object to use.
    client_manager : ClientManager
        The ClientManager object to be updated.

    Returns
    -------
    threading.Thread
        A thread that updates the ClientManager and handles the stop event.
    threading.Event
        An event that, when set, signals the thread to stop.
    """
    f_stop = threading.Event()
    thread = threading.Thread(
        target=_update_client_manager,
        args=(
            driver,
            client_manager,
            f_stop,
        ),
        daemon=True,
    )
    thread.start()

    return thread, f_stop


def _update_client_manager(
    driver: Driver,
    client_manager: ClientManager,
    f_stop: threading.Event,
) -> None:
    """Update the nodes list in the client manager."""
    # Loop until the driver is disconnected
    registered_nodes: Dict[int, GrpcDriverClientProxy] = {}
    while not f_stop.is_set():
        all_node_ids = set(driver.get_node_ids())
        dead_nodes = set(registered_nodes).difference(all_node_ids)
        new_nodes = all_node_ids.difference(registered_nodes)

        # Unregister dead nodes
        for node_id in dead_nodes:
            client_proxy = registered_nodes[node_id]
            client_manager.unregister(client_proxy)
            del registered_nodes[node_id]

        # Register new nodes
        for node_id in new_nodes:
            client_proxy = GrpcDriverClientProxy(
                node_id=node_id,
                driver=driver.grpc_driver_helper,  # type: ignore
                anonymous=False,
                run_id=driver.run_id,  # type: ignore
            )
            if client_manager.register(client_proxy):
                registered_nodes[node_id] = client_proxy
            else:
                raise RuntimeError("Could not register node.")

        # Sleep for 3 seconds
        if not f_stop.is_set():
            f_stop.wait(3)
