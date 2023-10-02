from typing import List, Tuple, Union

import numpy as np
from flwr.common import Metrics
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


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
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("here and nothing is breaking!!!")
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


class FedNova(FedAvg):
    """FedNova"""

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):

        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = sum(local_tau)
        # print("--------------------Called Stratgegy--------------------")
        # print("tau_eff", tau_eff)

        aggregate_parameters = []

        for client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            # print("params", client, len(params), type(params))
            scale = res.metrics["weight"]*(tau_eff/res.metrics["local_norm"])
            aggregate_parameters.append((params, scale))
            # for i in range(len(params)):
            #     params[i] = scale*params[i]
            #     print("-----type params -----------", type(params[i]), params[i].shape)

            # aggregate_parameters.append(params)

        # aggregate_parameters = np.array(aggregate_parameters,dtype=object)      # shape = (num_clients, num_parameters, ?)
        # aggregate_parameters = np.sum(aggregate_parameters,axis=0)              # shape = (num_parameters, ?)
        # print("------- Aggregate params shape ---------", aggregate_parameters.shape)
        # aggregate_parameters = ndarrays_to_parameters(list(aggregate_parameters))
        params_agg = ndarrays_to_parameters(aggregate(aggregate_parameters))
        # print("------- Aggregate params success ---------", type(params_agg))
        return params_agg, {}


from functools import reduce
from typing import List, Tuple
from flwr.common import NDArray, NDArrays


def aggregate(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training


    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * scale for layer in weights] for weights, scale in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
