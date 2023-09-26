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
    def select(self, valid_client: int) -> bool:
        return [str(result) for result in range(0, valid_client)]


class Fedmeta_client_manager(SimpleClientManager):
    def __init__(self, valid_client, **kwargs):
        super().__init__(**kwargs)
        self.valid_client = valid_client

    def sample(
        self,
        num_clients: int,
        server_round: Optional[int] = None,
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
            available_cids = criterion.select(self.valid_client)

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        if server_round is not None:
            random.seed(server_round)
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
