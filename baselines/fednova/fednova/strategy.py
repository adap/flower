from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
	Metrics,
	NDArray,
	NDArrays,
	Parameters,
	ndarrays_to_parameters,
	parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from omegaconf import DictConfig


class FedNova(FedAvg):
	"""FedNova"""

	def __init__(self, exp_config: DictConfig, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.global_momentum_buffer: List[NDArray] = []
		# self.old_parameter_init: List[NDArray] = parameters_to_ndarrays(self.initial_parameters)
		self.global_parameters: List[NDArray] = parameters_to_ndarrays(self.initial_parameters)

		self.exp_config = exp_config
		self.lr = exp_config.optimizer.lr
		self.gmf = exp_config.optimizer.gmf
		self.best_test_acc = 0.0

	def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
					  failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):

		if not results:
			return None, {}

		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}

		local_tau = [res.metrics["tau"] for _, res in results]
		tau_eff = sum(local_tau)

		aggregate_parameters = []

		for client, res in results:
			params = parameters_to_ndarrays(res.parameters)
			# scale = res.metrics["weight"] * (tau_eff / res.metrics["local_norm"])
			scale = res.metrics["weight"]
			aggregate_parameters.append((params, scale))

		agg_cum_gradient = aggregate(aggregate_parameters)
		self.update_server_params(agg_cum_gradient)

		return ndarrays_to_parameters(self.global_parameters), {}

	def update_server_params(self, cum_grad: NDArrays):

		for i, layer_cum_grad in enumerate(cum_grad):

			if self.gmf != 0:

				# check if it's the first round of aggregation, if so, initialize the global momentum buffer
				if len(self.global_momentum_buffer) == 0:
					buf = layer_cum_grad / self.lr
					self.global_momentum_buffer.append(buf)

				else:
					self.global_momentum_buffer[i] *= self.gmf
					self.global_momentum_buffer[i] += layer_cum_grad / self.lr

				self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr

			else:
				self.global_parameters[i] -= layer_cum_grad

	def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:

		"""Overide default evaluate method to save model parameters"""

		if self.evaluate_fn is None:
			# No evaluation function provided
			return None

		parameters_ndarrays = parameters_to_ndarrays(parameters)
		eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

		if eval_res is None:
			return None
		else:
			loss, metrics = eval_res
			accuracy = metrics["accuracy"]

			# print("----------- Round: {} Test Loss: {} Test Accuracy : {}--------------------".format(server_round, loss, accuracy))

			if accuracy > self.best_test_acc:
				self.best_test_acc = accuracy

				# Save model parameters and state
				if server_round == 0:
					return None
				else:
					np.savez(
						f"{self.exp_config.checkpoint_path}bestModel_{self.exp_config.exp_name}_{self.exp_config.strategy}_varEpochs_{self.exp_config.var_local_epochs}.npz",
						self.global_parameters, [loss, self.best_test_acc], self.global_momentum_buffer)

				log(INFO, "Model saved with Best Test accuracy: {}".format(self.best_test_acc))

			return loss, metrics


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
	return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
