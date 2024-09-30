# from typing import List, Tuple

from typing import Callable, Dict, List, Optional, Tuple

import os
import numpy as np
import flwr as fl
from flwr.common import Metrics
from flwr.server.utils import tensorboard
from flwr.server.strategy import FedAvg
from collections import OrderedDict
import torch
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flwr_logs")

# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# @tensorboard(logdir=LOGDIR)
# class SaveModelStrategyNpz(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[fl.common.Weights]:
#         weights = super().aggregate_fit(rnd, results, failures)
#         if weights is not None:
#             # Save weights
#             print(f"Saving round {rnd} weights...")
#             np.savez(f"round-{rnd}-weights.npz", *weights)
#         return weights

# @tensorboard(logdir=LOGDIR)
# class SaveModelStrategyPt(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[fl.common.Weights]:
#         aggregated_parameters = super().aggregate_fit(rnd, results, failures)
#         if aggregated_parameters is not None:
#             # aggregated_parameters = fl.common.para(aggregated_parameters)
#             aggregated_parameters = fl.common.parameters_to_weights(aggregated_parameters)
#             # Convert `Parameters` to `List[np.ndarray]`
#             aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(aggregated_parameters)
#
#             # Load PyTorch model
#             net = Generator().to(DEVICE)
#
#             # Convert `List[np.ndarray]` to PyTorch`state_dict`
#             params_dict = zip(net.state_dict().keys(), aggregated_weights)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             net.load_state_dict(state_dict, strict=True)
#             torch.save(net.state_dict(), 'best.pt')
#             # TODO Save PyTorch `state_dict` as `.pt`
#         return aggregated_weights


strategy = tensorboard(logdir=LOGDIR)(FedAvg)()



# Start Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

