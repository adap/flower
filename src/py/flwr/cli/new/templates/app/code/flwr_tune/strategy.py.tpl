"""$project_name: A Flower / FlowerTune app."""

from logging import INFO, WARN
from typing import List, Tuple

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    log,
    FitRes,
    Parameters,
    parameters_to_ndarrays,
)


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_fit(self, parameters: Parameters, **kwargs):
        """Configure the next round of training."""
        return_clients = super().configure_fit(parameters=parameters, **kwargs)

        # Test communication costs
        num_clients = len(return_clients)
        test_communication_costs(parameters, num_clients)

        return return_clients

    def aggregate_fit(self, results: List[Tuple[ClientProxy, FitRes]], **kwargs):
        """Aggregate fit results using weighted average."""
        # Test communication costs
        num_clients = len(results)
        test_communication_costs(results[0][1].parameters, num_clients)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(results=results, **kwargs)

        return parameters_aggregated, metrics_aggregated


def test_communication_costs(parameters, num_clients):
    """Test communication costs per FL round."""
    weights = parameters_to_ndarrays(parameters)

    size_in_bytes = sum([ele.nbytes for ele in weights])
    comm_cost = 2 * num_clients * size_in_bytes / 1024**2
    log(INFO, f"Communication costs per round: {comm_cost} MB")

    if comm_cost > 500:
        log(WARN,
            "The total communication costs per round exceed 500 MB. "
            "Please consider reducing it if you plan to participate "
            "FlowerTune LLM Leaderboard.",
            )
