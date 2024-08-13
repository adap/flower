"""$project_name: A Flower / FlowerTune app."""

from logging import INFO, WARN
from typing import List, Tuple, Union

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    log,
    FitIns,
    FitRes,
    Parameters,
    parameters_to_ndarrays,
)


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation."""
    def configure_fit(self, parameters: Parameters, **kwargs):
        """Configure the next round of training."""
        return_clients = super().configure_fit(parameters=parameters, **kwargs)

        # Test communication costs
        fit_ins_list = [fit_ins for _, fit_ins in return_clients]
        test_communication_costs(fit_ins_list)

        return return_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results using weighted average."""
        # Test communication costs
        fit_res_list = [fit_res for _, fit_res in results]
        test_communication_costs(fit_res_list)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated


def test_communication_costs(fit_list: List[Union[FitIns, FitRes]]):
    """Test communication costs per FL round."""
    def compute_bytes(weights):
        return sum([ele.nbytes for ele in weights])

    size_bytes_list = [compute_bytes(parameters_to_ndarrays(fit_ele.parameters)) for fit_ele in fit_list]
    comm_cost = 2 * sum(size_bytes_list) / 1024**2
    log(INFO, f"Communication costs per round: {comm_cost} MB")

    if comm_cost > 500:
        log(WARN,
            "The total communication costs per round exceed 500 MB. "
            "Please consider reducing it if you plan to participate "
            "FlowerTune LLM Leaderboard.",
            )
