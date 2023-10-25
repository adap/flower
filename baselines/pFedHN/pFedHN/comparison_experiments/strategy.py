"""Strategy for comparison Federated Learning algorithms with pFedHN."""

import json
from logging import DEBUG
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

results_li = []


# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
class LogResultsStrategy(fl.server.strategy.fedavg.FedAvg):
    """Federated strategy for aggregating the weights of the clients as well as logging.

    them.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the weights of the clients and log them."""
        total_loss = 0.0
        total_acc = 0.0
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        for _client, fit_res in results:
            total_loss += float(fit_res.metrics["test_loss"])
            total_acc += float(fit_res.metrics["test_acc"])
        total_samp = len(results)
        log(
            DEBUG,
            f"TestLoss: {total_loss/total_samp} || TestAcc: {total_acc/total_samp}",
        )
        results_dict = {
            "round": server_round,
            "loss": float(total_loss / len(results)),
            "accuracies": float(total_acc / len(results)),
        }
        results_li.append(results_dict)
        with open("res.json", "w") as jsonfile:
            json.dump(results_li, jsonfile)

        return parameters_aggregated, {}
