"""FedAvg with the same clients used for both training and evaluation."""

from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgSameClients(FedAvg):
    """FedAvg that samples clients for each round only once (the same clients
    are used for training and testing round n)

    It does not mean that the same client are used in each round. It used just the same clients
    (with different parts of their data) in round i.

    It assumes that there is no different function for evaluation - on_evaluate_config_fn
    (it's ignored).
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self._current_round_fit_clients_fits_list: List[Tuple[ClientProxy, FitIns]] = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self._current_round_fit_clients_fits_list = super().configure_fit(
            server_round, parameters, client_manager
        )
        # Return client/config pairs
        return self._current_round_fit_clients_fits_list

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Keep the fraction_settings for consistency reasons
        if self.fraction_evaluate == 0.0:
            return []
        evaluate_config = []
        for tuple_client_proxy_fit_ins in self._current_round_fit_clients_fits_list:
            eval_ins = EvaluateIns(
                tuple_client_proxy_fit_ins[1].parameters,
                tuple_client_proxy_fit_ins[1].config,
            )
            evaluate_config.append((tuple_client_proxy_fit_ins[0], eval_ins))
        return evaluate_config
