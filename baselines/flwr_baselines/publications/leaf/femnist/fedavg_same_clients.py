from typing import List, Tuple

from flwr.common import Parameters, FitIns, EvaluateIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgSameClients(FedAvg):
    """FedAvg that samples clients for each round only once (the same clinets are used for training and testing round n)

    It does not mean that the same client are used in each round. It used just the same clients (with different parts of their data) in round i.

    It assumes that there is no different function for evaluation - on_evaluate_config_fn (it's ignored).
    """

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self._current_round_fit_clients_fits_list = [(client, fit_ins) for client in clients]
        # Return client/config pairs
        return self._current_round_fit_clients_fits_list

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Keep the fraction_settings for consistency reasons
        if self.fraction_evaluate == 0.0:
            return []

        return self._current_round_fit_clients_fits_list
