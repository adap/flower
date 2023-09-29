"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING, INFO
from collections import OrderedDict

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_manager import ClientManager
from Fedmeta_client_manager import evaluate_client_Criterion
import numpy as np
import torch
from functools import reduce
from models import CNN_network, StackedLSTM


from flwr.common.logger import log
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateRes,
    Metrics,
    FitIns,
    EvaluateIns,
    NDArrays,
)

import wandb

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="SoR",
#
# )


def fedmeta_update_meta_sgd(net, alpha, beta, weights_results, gradients_aggregated):
    params_dict = zip(net.state_dict().keys(), weights_results)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(alpha), lr=beta, weight_decay=0.0001)
    for params, grad_ins, alphas in zip(net.parameters(), gradients_aggregated, alpha):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
        alphas.grad = torch.tensor(grad_ins).to(params.dtype)
    optimizer.step()
    optimizer.zero_grad()
    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return weights_prime, alpha


def fedmeta_update_maml(net, beta, weights_results, gradients_aggregated):
    params_dict = zip(net.state_dict().keys(), weights_results)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=beta, weight_decay=0.0001)
    for params, grad_ins in zip(net.parameters(), gradients_aggregated):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
    optimizer.step()
    optimizer.zero_grad()
    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return weights_prime

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    correct = [num_examples * m["correct"] for num_examples, m in metrics]
    # correct = [m["correct"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(correct) / sum(examples)}


def aggregate_grad(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute gradients average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_gradients = [
        [layer * num_examples for layer in gradients] for gradients, num_examples in results
    ]

    # Compute average weights of each layer
    grdients_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_gradients)
    ]

    return grdients_prime


class FedMeta(FedAvg):
    def __init__(self, alpha, beta, data, algo, **kwargs):
        super().__init__(**kwargs)
        self.algo = algo
        self.data = data
        if self.data == 'femnist':
            self.net = CNN_network()
        elif self.data == 'shakespeare':
            self.net = StackedLSTM()
        self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.full_like(p, alpha)) for p in self.net.parameters()])
        self.beta = beta

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # alpha_list = [param.data for param in self.alpha]
        config = {"alpha" : self.alpha, "algo": self.algo, "data": self.data}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {"alpha" : self.alpha, "algo": self.algo, "data":self.data}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            server_round=server_round,
            min_num_clients=min_num_clients,
            criterion=evaluate_client_Criterion(self.min_evaluate_clients),
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        if self.algo == 'fedmeta(maml)':
            grads_results = [
                (fit_res.metrics['grads'], fit_res.num_examples)
                for _, fit_res in results
            ]
            gradients_aggregated = aggregate_grad(grads_results)
            weights_prime = fedmeta_update_maml(self.net, self.beta, weights_results[0][0], gradients_aggregated)
            parameters_aggregated = ndarrays_to_parameters(weights_prime)

        elif self.algo == 'fedmeta(meta-sgd)':
            grads_results = [
                (fit_res.metrics['grads'], fit_res.num_examples)
                for _, fit_res in results
            ]
            gradients_aggregated = aggregate_grad(grads_results)
            weights_prime, update_alpha = fedmeta_update_meta_sgd(self.net, self.alpha, self.beta, weights_results[0][0], gradients_aggregated)
            self.alpha = update_alpha
            parameters_aggregated = ndarrays_to_parameters(weights_prime)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        weight_loss = sum([evaluate_res.metrics['loss'] * evaluate_res.num_examples for _, evaluate_res in results]) / sum(
            [evaluate_res.num_examples for _, evaluate_res in results])
        # wandb.log({"Training Loss": weight_loss}, step=server_round)
        log(INFO, f'Training Loss : {weight_loss}')

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            # wandb.log({"Test_Accuracy ": round(metrics_aggregated['accuracy'] * 100, 3)}, step=server_round)
            log(INFO, f'Test Accuracy : {round(metrics_aggregated["accuracy"] * 100, 3)}')

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
