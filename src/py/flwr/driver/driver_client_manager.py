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
"""Flower DriverClientManager."""


import random
import time
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log
from flwr.proto import driver_pb2
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from .driver import Driver
from .driver_client_proxy import DriverClientProxy


class DriverClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self.clients: Dict[str, ClientProxy] = {}

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        self._update_nodes()
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        raise NotImplementedError("DriverClientManager.register is not implemented")

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        raise NotImplementedError("DriverClientManager.unregister is not implemented")

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        self._update_nodes()
        return self.clients

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available."""
        start_time = time.time()
        while time.time() < start_time + timeout:
            self._update_nodes()
            if len(self.clients) >= num_clients:
                return True
            time.sleep(1)
        return False

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        available_cids = list(self.clients)

        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

    def _update_nodes(self) -> None:
        """Update the nodes list in the client manager.

        This method communicates with the associated driver to get all node ids. Each
        node id is then converted into a `DriverClientProxy` instance and stored in the
        `clients` dictionary with node id as key.
        """
        get_nodes_res = self.driver.get_nodes(req=driver_pb2.GetNodesRequest())
        all_node_ids = get_nodes_res.node_ids
        for node_id in all_node_ids:
            self.clients[str(node_id)] = DriverClientProxy(
                node_id=node_id,
                driver=self.driver,
                anonymous=False,
            )
