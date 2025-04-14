"""$project_name: A Flower / FlowerTune app."""

from io import BytesIO
from logging import INFO, WARN
from typing import List, Tuple, Union

from flwr.common import FitIns, FitRes, Parameters, log, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.
    
    This class behaves just like FedAvg but also tracks the communication
    costs associated with `fit` over FL rounds.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training."""
        return_clients = super().configure_fit(server_round, parameters, client_manager)

        # Test communication costs
        fit_ins_list = [fit_ins for _, fit_ins in return_clients]
        self.comm_tracker.track(fit_ins_list)

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
        self.comm_tracker.track(fit_res_list)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        return parameters_aggregated, metrics_aggregated


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""
    def __init__(self):
        self.curr_comm_cost = 0.0

    @staticmethod
    def _compute_bytes(parameters):
        return sum([BytesIO(t).getbuffer().nbytes for t in parameters.tensors])

    def track(self, fit_list: List[Union[FitIns, FitRes]]):
        size_bytes_list = [
            self._compute_bytes(fit_ele.parameters)
            for fit_ele in fit_list
        ]
        comm_cost = sum(size_bytes_list) / 1024**2

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
