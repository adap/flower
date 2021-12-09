# Copyright 2021 Adap GmbH. All Rights Reserved.
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
"""Flower PriorityClientManager."""


import numpy as np
from typing import List, Optional

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.client_manager import SimpleClientManager
from flwr.common.typing import PropertiesIns


class PriorityClientManager(SimpleClientManager):
    """Provides a pool of available clients based on their priority."""

    def __init__(self, criterion: Criterion = None) -> None:
        super().__init__()
        self.criterion = criterion

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)

        # Use the to check if clients have `priority` property client
        if criterion is not None:
            self.criterion = criterion
            available_cids = [
                cid
                for cid in available_cids
                if self.criterion.select(self.clients[cid])
            ]

        # Sends an empty `config` dictionary to all clients, encapsulated in PropertiesIns.
        # This could be used, for example, to send a different configuration based on round, etc...
        ins = PropertiesIns(config={})
        num_samples_vect = [
            self.clients[cid].get_properties(ins=ins).properties["priority"]
            for cid in available_cids
        ]
        prob = np.asarray(num_samples_vect) / np.sum(num_samples_vect)
        sampled_cids = np.random.choice(
            available_cids, size=num_clients, replace=False, p=prob
        )

        return [self.clients[cid] for cid in sampled_cids]
