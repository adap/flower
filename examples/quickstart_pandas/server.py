import pickle
from typing import Callable, List, Tuple

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy


class FedAnalytics(Strategy):
    def __init__(
        self, compute_fns: List[Callable] = None, col_names: List[str] = None
    ) -> None:
        super().__init__()
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )

    def initialize_parameters(self, client_manager=None):
        return self.parameters

    def configure_fit(self, server_round, parameters, client_manager):
        # Tell fit what to do - this isnt really doing anything currently
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        # Get results from fit
        # Convert results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]
        # Hacking around to load back data from parameters
        metrics_aggregated = {}
        for index, arr in enumerate(values_aggregated):
            i = 0
            metrics_aggregated[index] = {}
            metrics_aggregated[index] = arr
        print(metrics_aggregated)
        return [], metrics_aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        pass

    def aggregate_evaluate(self, server_round, results, failures):
        pass

    def evaluate(self, data, parameters):
        pass


strategy = FedAnalytics()

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)
