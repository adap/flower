"""FjORD strategy."""

from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .client import FJORD_CONFIG_TYPE
from .utils.logger import Logger


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate using weighted average based on number of samples.

    :param metrics: List of tuples (num_examples, metrics)
    :return: Aggregated metrics
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = np.array([num_examples * m["accuracy"] for num_examples, m in metrics])
    examples = np.array([num_examples for num_examples, _ in metrics])

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": accuracies.sum() / examples.sum()}


def get_p_layer_updates(
    p: float,
    layer_updates: List[np.ndarray],
    num_examples: List[int],
    p_max_s: List[float],
) -> Tuple[List[np.ndarray], int]:
    """Get layer updates for given p width.

    :param p: p-value
    :param layer_updates: list of layer updates from clients
    :param num_examples: list of number of examples from clients
    :param p_max_s: list of p_max values from clients
    """
    # get layers that were updated for given p
    # i.e., for the clients with p_max >= p
    layer_updates_p = [
        layer_update
        for p_max, layer_update in zip(p_max_s, layer_updates)
        if p_max >= p
    ]
    num_examples_p = sum(n for p_max, n in zip(p_max_s, num_examples) if p_max >= p)
    return layer_updates_p, num_examples_p


def fjord_average(  # pylint: disable=too-many-arguments
    i: int,
    layer_updates: List[np.ndarray],
    num_examples: List[int],
    p_max_s: List[float],
    p_s: List[float],
    fjord_config: FJORD_CONFIG_TYPE,
    original_parameters: List[np.ndarray],
) -> np.ndarray:
    """Compute average per layer for given updates.

    :param i: index of the layer
    :param layer_updates: list of layer updates from clients
    :param num_examples: list of number of examples from clients
    :param p_max_s: list of p_max values from clients
    :param p_s: list of p values
    :param fjord_config: fjord config
    :param original_parameters: original model parameters
    :return: average of layer
    """
    # if no client updated the given part of the model,
    # reuse previous parameters
    update = deepcopy(original_parameters[i])

    # BatchNorm2d layers, only average over the p_max_s
    # that are greater than corresponding p of the layer
    # i.e., only update the layers that were updated
    if fjord_config["layer_p"][i] is not None:
        p = fjord_config["layer_p"][i]
        layer_updates_p, num_examples_p = get_p_layer_updates(
            p, layer_updates, num_examples, p_max_s
        )
        if len(layer_updates_p) == 0:
            return update

        assert num_examples_p > 0
        return reduce(np.add, layer_updates_p) / num_examples_p
    if fjord_config["layer"][i] in ["ODLinear", "ODConv2d", "ODBatchNorm2d"]:
        # perform nested updates
        for p in p_s[::-1]:
            layer_updates_p, num_examples_p = get_p_layer_updates(
                p, layer_updates, num_examples, p_max_s
            )
            if len(layer_updates_p) == 0:
                continue
            in_dim = (
                int(fjord_config[p][i]["in_dim"])
                if fjord_config[p][i]["in_dim"]
                else None
            )
            out_dim = (
                int(fjord_config[p][i]["out_dim"])
                if fjord_config[p][i]["out_dim"]
                else None
            )
            assert num_examples_p > 0
            # check whether the parameter to update is bias or weight
            if len(update.shape) == 1:
                # bias or ODBatchNorm2d
                layer_updates_p = [
                    layer_update[:out_dim] for layer_update in layer_updates_p
                ]
                update[:out_dim] = reduce(np.add, layer_updates_p) / num_examples_p
            else:
                # weight
                layer_updates_p = [
                    layer_update[:out_dim, :in_dim] for layer_update in layer_updates_p
                ]
                update[:out_dim, :in_dim] = (
                    reduce(np.add, layer_updates_p) / num_examples_p
                )
        return update

    raise ValueError(f"Unsupported layer {fjord_config['layer'][i]}")


def aggregate(
    results: List[Tuple[NDArrays, int, float, List[float], FJORD_CONFIG_TYPE]],
    original_parameters,
) -> NDArrays:
    """Compute weighted average.

    :param results: list of tuples (layer_updates, num_examples, p_max, p_s)
    :param original_parameters: original model parameters
    :return: weighted average of layer updates
    """
    # Create a list of weights, each multiplied
    # by the related number of examples
    weights = [
        [param * num_examples for param in params]
        for params, num_examples, _, _, _ in results
    ]
    p_max_s = [p_max for _, _, p_max, _, _ in results]

    # Calculate the total number of examples used during training
    num_examples = [num_examples for _, num_examples, _, _, _ in results]
    p_s = results[0][3]
    fjord_config = results[0][4]

    weights_prime: NDArrays = [
        fjord_average(
            i,
            layer_updates,
            num_examples,
            p_max_s,
            p_s,
            fjord_config,
            original_parameters,
        )
        for i, layer_updates in enumerate(zip(*weights))
    ]
    return weights_prime


class FjORDFedAVG(FedAvg):
    """FedAvg strategy with FjORD aggregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        Logger.get().info(f"Aggregating for global round {server_round}")
        # Convert results
        weights_results: List[
            Tuple[NDArrays, int, float, List[float], FJORD_CONFIG_TYPE]
        ] = [
            (  # type: ignore
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                fit_res.metrics["max_p"],
                fit_res.metrics["p_s"],
                fit_res.metrics["fjord_config"],
            )
            for _, fit_res in results
        ]

        p_max_values_str = ", ".join([str(val[2]) for val in weights_results])
        Logger.get().info(f"\t - p_max values: {p_max_values_str}")

        # all clients start with the same model
        for _, fit_res in results:
            original_parameters = fit_res.metrics["original_parameters"]
            break

        training_losses_str = ", ".join(
            [str(fit_res.metrics["loss"]) for _, fit_res in results]
        )
        Logger.get().info(f"\t - train losses: {training_losses_str}")

        agg = aggregate(weights_results, original_parameters)

        parameters_aggregated = ndarrays_to_parameters(agg)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            Logger.get().warn("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
