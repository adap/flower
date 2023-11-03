import enum
import flwr as fl
from flwr.server.strategy.fedhenn import FedHeNN
from logging import WARNING, DEBUG
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from dataset import load_mnist_data_partition
from model_mnist import Net0, Net1, Net2, Net3
from collections import OrderedDict
import torch
from similarity_utils import cka_torch, gram_linear_torch, gram_linear
import numpy as np
import json
from json_utils import NumpyArrayEncoder


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""
models_dict = {"model_a": Net0, "model_b": Net1, "model_c": Net2, "model_d": Net3}


def compute_K_val(parameters: Parameters, model_type, dataloader):
    Net = model_type()
    params_dict = zip(Net.state_dict().keys(), parameters_to_weights(parameters))
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    Net.load_state_dict(state_dict, strict=True)
    for images_RAD, _ in dataloader:
        intermediate_activation, _ = Net(images_RAD)

    final_array = np.array(gram_linear(intermediate_activation.detach().numpy()))
    return final_array


class custom_FedHeNN(FedHeNN):

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):

            initial_parameters_ = [
                weights_to_parameters(
                    weights=initial_parameter, tensor_type=f"model_{letter}"
                )
                for initial_parameter, letter in zip(initial_parameters, list("abcd"))
            ]

        return initial_parameters_

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        trainloader_RAD, testloader_RAD, num_examples_RAD = load_mnist_data_partition(
            batch_size=32,
            partitions=5,
            RAD=True,
            subsample_RAD=True,
            use_cuda=False,
            input_seed=rnd,
        )
        if isinstance(parameters, list):
            # different model parameters for different clients
            fit_ins = []
            K_final = np.sum(
                np.array(
                    [
                        compute_K_val(
                            parameter,
                            models_dict[parameter.tensor_type],
                            trainloader_RAD,
                        )
                        for parameter in parameters
                    ]
                ),
                axis=0,
            )

            for parameter in parameters:
                config = {}
                if self.on_fit_config_fn is not None:
                    # Custom fit config function provided
                    config = self.on_fit_config_fn(rnd)
                config["K_final"] = json.dumps(K_final, cls=NumpyArrayEncoder)
                config["tensor_type"] = parameter.tensor_type
                fit_ins.append(FitIns(parameter, config))

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_in) for client, fit_in in zip(clients, fit_ins)]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_eval == 0.0:
            return []

        # Parameters and config
        trainloader_RAD, testloader_RAD, num_examples_RAD = load_mnist_data_partition(
            batch_size=32,
            partitions=5,
            RAD=True,
            subsample_RAD=True,
            use_cuda=False,
            input_seed=rnd,
        )
        if isinstance(parameters, list):
            evaluate_ins = []
            K_final = np.sum(
                np.array(
                    [
                        compute_K_val(
                            parameter,
                            models_dict[parameter.tensor_type],
                            testloader_RAD,
                        )
                        for parameter in parameters
                    ]
                ),
                axis=0,
            )
            for parameter in parameters:
                config = {}
                if self.on_evaluate_config_fn is not None:
                    # Custom evaluation config function provided
                    config = self.on_evaluate_config_fn(rnd)
                config["K_final"] = json.dumps(K_final, cls=NumpyArrayEncoder)
                config["tensor_type"] = parameter.tensor_type
                evaluate_ins.append(EvaluateIns(parameter, config))

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [
            (client, evaluate_in) for client, evaluate_in in zip(clients, evaluate_ins)
        ]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.metrics["tensor_type"])
            for _, fit_res in results
        ]
        parameters_results = [
            weights_to_parameters(weight, model_type)
            for (weight, model_type) in weights_results
        ]

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {
            (
                fit_res.metrics["tensor_type"],
                fit_res.metrics["loss1"],
                fit_res.metrics["loss2"],
            )
            for _, fit_res in results
        }
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif rnd == 1:
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")
        log(DEBUG, f"FIT: tensor_type, loss1, loss2")
        log(DEBUG, f"aggregate fit logs: {metrics_aggregated}")

        return parameters_results, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (evaluate_res.num_examples, evaluate_res.loss)
        #         for _, evaluate_res in results
        #     ]
        # )

        loss_aggregated = [evaluate_res.loss for _, evaluate_res in results]
        metrics_aggregated = {}
        metrics = {
            (
                evaluate_res.metrics["tensor_type"],
                evaluate_res.metrics["accuracy"],
                evaluate_res.metrics["cka_score"],
                evaluate_res.metrics["loss"],
            )
            for _, evaluate_res in results
        }
        # Aggregate custom metrics if aggregation fn was provided
        # metrics_aggregated = {}
        # if self.evaluate_metrics_aggregation_fn:
        #     eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        # elif rnd == 1:
        #     log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        log(DEBUG, f"EVAL: tensor_type, accuracy, cka_score, loss")
        log(DEBUG, f"aggregate eval logs: {metrics}")
        return loss_aggregated, metrics_aggregated
