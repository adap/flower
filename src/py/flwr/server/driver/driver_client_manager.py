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


import threading
import uuid
from typing import Dict, List, Optional, Set

from ..client_manager import ClientManager
from ..client_proxy import ClientProxy
from ..criterion import Criterion


class DriverClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self.cids_to_ids: Dict[str, int] = {}
        self.ids_to_cps: Dict[int, ClientProxy] = {}

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.ids_to_cps)

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
        if client.cid in self.cids_to_ids:
            return False

        # Generate random integer ID
        random_client_id: int = uuid.uuid1().int >> 64

        # Store cid, id, and ClientProxy
        self.cids_to_ids[client.cid] = random_client_id
        self.ids_to_cps[random_client_id] = client

        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.cids_to_ids:
            del self.ids_to_cps[self.cids_to_ids[client.cid]]
            del self.cids_to_ids[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all_ids(self) -> Set[int]:
        """Return all available client ids.

        Returns
        -------
        ids : Set[int]
            The IDs of all currently available clients.
        """
        return set(self.ids_to_cps)

    # --- Unimplemented methods -----------------------------------------------

    def all(self) -> Dict[str, ClientProxy]:
        """Not implemented."""
        raise NotImplementedError()

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Not implemented."""
        raise NotImplementedError()

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Not implemented."""
        raise NotImplementedError()
