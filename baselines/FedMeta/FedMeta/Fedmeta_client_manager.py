from flwr.server.client_manager import SimpleClientManager
from typing import List, Optional
from logging import INFO
from flwr.common.logger import log
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy
import random


class evaluate_client_Criterion(Criterion):
    def __init__(self, min_evaluate_clients):
        self.min_evaluate_clients = min_evaluate_clients

    """Criterion to select evaluate clients."""
    def select(self, clients_num: int) -> bool:
        return [str(result) for result in range(0, min(self.min_evaluate_clients, clients_num))]


class Fedmeta_client_manager(SimpleClientManager):
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
        if criterion is not None:
            available_cids = criterion.select(len(self.clients))

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
