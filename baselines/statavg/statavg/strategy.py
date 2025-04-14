"""StatAvg strategy."""

import json
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


# Custom class for implementing StatAvg (inherits properties from FedAvg)
class CustomStatAvg(fl.server.strategy.FedAvg):
    """StatAvg.

    The server receives the client local statistics. only at the 1st round, and
    generates the global statistics. It sends the global statistics via the config in
    configure_fit() method at 2nd round. Afterwards, it does nothing new, and simply
    follows FedAvg.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the attribute metrics_aggregated and initialize with arbitrary values
        # this will carry the global aggregated statistics
        self.metrics_aggregated = {"initialization": 0}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate both parameters and local statistics."""
        # thats to prevent aggregate_fit from making aggregations
        if server_round > 1:
            self.fit_metrics_aggregation_fn = None

        parameters_aggregated, self.metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        return parameters_aggregated, self.metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Build server's config file."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # ----------------- This part is custom -----------------------

        # The server will communicate the global statistics via the config

        # Add the global statistics to the config file
        if server_round <= 2:
            key = list(self.metrics_aggregated.keys())[0]
            value = self.metrics_aggregated[key]
            config[key] = value

        # -------------------------------------------------------------

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Calculate a weighted average of the clients' accuracy."""
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round}, aggr. client accuracy: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

    @staticmethod
    def get_average_statistics(
        fit_metrics: List[Tuple[int, Dict[str, int]]]
    ) -> Dict[str, int]:
        """Return the aggregated metrics.

        Will be passed to fit_metrics_aggregation_fn. It reads the received statistics
        from each client, and aggregates them. These global statistics will be saved a
        Dict. Afterwards, they will be sent to all clients.
        """
        stats = [json.loads(list(metrics.keys())[0]) for _, metrics in fit_metrics]
        mean = [np.array(s["mean"]) for s in stats]
        var = [np.array(s["var"]) for s in stats]
        num_examples = [n for n, metrics in fit_metrics]

        # Compute global mean and variance according to the paper
        mean_global = sum(m * n for m, n in zip(mean, num_examples)) / (
            sum(num_examples)
        )
        var_global = sum(
            n * v + n * (m - mean_global) ** 2
            for v, m, n in zip(var, mean, num_examples)
        ) / (sum(num_examples))

        # check the type (in case where the stats are floats)
        mean_global = (
            mean_global.tolist() if isinstance(mean_global, np.ndarray) else mean_global
        )
        var_global = (
            var_global.tolist() if isinstance(var_global, np.ndarray) else var_global
        )

        # Convert the global mean and variances to json then to Dict[str, Scalar]
        stats_global = {
            "mean_global": mean_global,
            "var_global": var_global,
        }
        json_stats_global = json.dumps(stats_global)

        # 0 in metrics_global is arbitrary and not used anywhere. Just for consistency.
        metrics_global = {json_stats_global: 0}

        return metrics_global


# Class for implementing FedAvg including the method aggregate_evaluate
class FedAvgAggrEv(fl.server.strategy.FedAvg):
    """FedAvg with aggregate_evaluate."""

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Calculate a weighted average of the clients' accuracy."""
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round}, aggr. client accuracy: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
