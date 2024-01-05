"""Strategy of Federated Learning.

Contains pFedHN/pFedHNPC and FedAvg strategies.
"""

import json
from logging import DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from pFedHN.utils import set_seed

set_seed(42)


# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
class pFedHN(fl.server.strategy.Strategy):
    """Federated strategy with pFedHN."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        fraction_fit: float = 0.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        evaluate_every: int = 1,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.evaluate_every = evaluate_every

    def __repr__(self) -> str:
        """Return the strategy name."""
        return "pFedHN"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        client_manager.num_available()
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for fit."""
        return min(1, num_available_clients), self.min_available_clients

    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Here only one client per round is taken as given in the pFedHN algorithm
        """
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        fit_configurations = []
        for _idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, {})))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Send final state_dict of the selected client to.

        the server for updating the Hypernetwork.

        These are recieved from only one client which was selected during configure_fit
        """
        _, fit_res = results[0]

        final_state_val = fit_res.parameters

        return final_state_val, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        This is the stage where all clients are being selected for Federated Evaluation
        . By default evaluation is done once in 30 rounds
        """
        if server_round % self.evaluate_every != 0:
            return []

        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        fit_configurations = []
        for _idx, client in enumerate(clients):
            fit_configurations.append((client, EvaluateIns(parameters, {})))

        return fit_configurations

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics.

        This function is responsible for the federated evaluation of various metircs
        recieved from all the clients selected in the configure_evalute function
        """
        if not results:
            return None, {}

        avg_loss = np.mean([evaluate_res.loss for _, evaluate_res in results])
        total_samples = sum(
            eres.metrics["total"] for _, eres in results  # type: ignore[misc]
        )
        total_correct = sum(
            eres.metrics["correct"] for _, eres in results  # type: ignore[misc]
        )
        avg_acc = total_correct / total_samples

        return avg_loss, {"avg_acc": avg_acc}  # type: ignore[return-value]

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


class FedAvgWithSaveJson(fl.server.strategy.FedAvg):
    """FedAvg strategy along with saving results in json file."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        evaluate_every,
        fraction_fit,
        min_fit_clients,
        fraction_evaluate,
        min_evaluate_clients,
        min_available_clients,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.evaluate_every = evaluate_every
        self.results_fedavg = []

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        By default all clients are selected for federated evaluation once in 30 rounds
        """
        if server_round % self.evaluate_every != 0:
            return []
        sample_size, min_num_clients = super().num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        fit_configurations = []
        for _idx, client in enumerate(clients):
            fit_configurations.append((client, EvaluateIns(parameters, {})))

        return fit_configurations

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}

        lossavg = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracies = [r.metrics["eval_acc"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        accavg: float = sum(accuracies) / sum(examples)  # type: ignore[arg-type]

        data = {
            "round": server_round,
            "avg_loss": lossavg,
            "avg_accuracy": accavg,
        }

        self.results_fedavg.append(data)

        with open("fedavg.json", "w", encoding="utf-8") as f:
            json.dump(self.results_fedavg, f)

        log(
            DEBUG,
            "AvgLoss: %.4f, AvgAcc: %.4f",
            lossavg,
            accavg,
        )

        return lossavg, {"avg_acc": accavg}  # type: ignore[return-value]
