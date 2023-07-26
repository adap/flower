from typing import List, Tuple
from flwr.common import EvaluateIns, FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from FedPer.utils.store_history_strategy import StoreHistoryStrategy


class StoreSelectedClientsStrategy(StoreHistoryStrategy):
    """Server FL selected client storage per training/evaluation round strategy implementation."""

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training and save the selected clients.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        result = super().configure_fit(rnd=rnd, parameters=parameters, client_manager=client_manager)

        if rnd not in self.hist["trn"].keys():
            self.hist["trn"][rnd] = {}

        self.hist["trn"][rnd]["selected_clients"] = [client.cid for client, _ in result]

        # Return client/config pairs
        return result

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation and save the selected clients.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        result = super().configure_evaluate(rnd=rnd, parameters=parameters, client_manager=client_manager)

        if rnd not in self.hist["tst"].keys():
            self.hist["tst"][rnd] = {}

        self.hist["tst"][rnd]["selected_clients"] = [client.cid for client, _ in result]

        # Return client/config pairs
        return result
