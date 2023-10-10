"""Strategy class for SCAFFOLD."""

from dataclasses import dataclass
from logging import WARNING

import torch
from flwr.common import (
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Union[int, Parameters]]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: Dict[str, Union[int, Parameters]]


FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        server_cv: List[torch.Tensor],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in server_cv]
        """Configure the next round of training."""
        config = {"server_cv": ndarrays_to_parameters(server_cv_np)}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config.update(self.on_fit_config_fn(server_round))
        fit_ins = FitIns(
            parameters, config
        )  # this FitIns has different types compared to the default

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Convert client cvs to ndarrays
        client_cv_updates = [
            parameters_to_ndarrays(fit_res.metrics["server_update_c"])
            for _, fit_res in results
        ]
        # zip client cvs and num_examples
        client_cv_updates = list(
            zip(client_cv_updates, [fit_res.num_examples for _, fit_res in results])
        )
        aggregated_cv_update = aggregate(client_cv_updates)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated, aggregated_cv_update
