from typing import List, Tuple, Union, Optional
import numpy as np
from flwr.common import Metrics
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from functools import reduce
from omegaconf import DictConfig
from flwr.common import (
	FitIns,
	FitRes,
	NDArrays,
	NDArray,
	Parameters,
	ndarrays_to_parameters,
	parameters_to_ndarrays,
)


class FedNova(FedAvg):
	"""FedNova"""

	def __init__(self, exp_config: DictConfig, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.global_momentum_buffer: List[NDArray] = []
		self.old_parameter_init: List[NDArray] = parameters_to_ndarrays(self.initial_parameters)
		self.global_parameters: List[NDArray] = []

		self.exp_config = exp_config
		self.lr = exp_config.optimizer.lr
		self.gmf = exp_config.optimizer.gmf


	def configure_fit(self, server_round: int, parameters: Parameters,
					  client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:

		"""Configure the next round of training."""

		# Adjust learning rate based on server rounds
		config = {}

		if self.exp_config.lr_scheduling:
			if server_round == int(self.exp_config.num_rounds / 2):
				self.lr = self.exp_config.optimizer.lr / 10

			elif server_round == int(self.exp_config.num_rounds * 0.75):
				self.lr = self.exp_config.optimizer.lr / 100

		config["lr"] = self.lr
		config["server_round"] = server_round

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
			scale = res.metrics["weight"] * (tau_eff / res.metrics["local_norm"])
			aggregate_parameters.append((params, scale))

		agg_cum_gradient = aggregate(aggregate_parameters)
		self.global_parameters = self.update_server_params(agg_cum_gradient)

		return ndarrays_to_parameters(self.global_parameters), {}

	def update_server_params(self, cum_grad: NDArrays):
		updated_params = []

		for i, layer_cum_grad in enumerate(cum_grad):

			if self.gmf != 0:

				# check if it's the first round of aggregation, if so, initialize the global momentum buffer
				if self.global_momentum_buffer is None:
					buf = layer_cum_grad / self.lr
					self.global_momentum_buffer.append(buf)

				else:
					self.global_momentum_buffer[i] *= self.gmf
					self.global_momentum_buffer[i] += layer_cum_grad / self.lr

				self.old_parameter_init[i] -= self.global_momentum_buffer[i] * self.lr

			else:
				self.old_parameter_init[i] -= layer_cum_grad

			updated_params.append(self.old_parameter_init[i])

		return updated_params



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
