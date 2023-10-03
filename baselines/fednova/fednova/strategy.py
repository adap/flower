from typing import List, Tuple, Union, Optional, Dict
import numpy as np
from flwr.common import Metrics
from flwr.common.typing import FitRes, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from omegaconf import DictConfig
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import INFO, WARNING

from flwr.common import (
	FitIns,
	NDArrays,
	NDArray,
	Parameters,
	ndarrays_to_parameters,
	parameters_to_ndarrays
)


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
		self.best_acc = 0.0

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

	def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:

		"""Edit Aggregate evaluation method to allow checkpointing of model parameters
		for best evaluation accuracy"""

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

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.evaluate_metrics_aggregation_fn:
			eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
			metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

			if metrics_aggregated.get("accuracy", 0.0) > self.best_acc:
				self.best_acc = metrics_aggregated.get("accuracy", 0.0)
				# Save model parameters and state

				np.savez(f"{self.exp_config.checkpoint_path}best_model_{server_round}.npz",
						 self.global_parameters, [self.best_acc], self.global_momentum_buffer)

				log(INFO, "Best model saved")

		elif server_round == 1:  # Only log this warning once
			log(WARNING, "No evaluate_metrics_aggregation_fn provided")

		return loss_aggregated, metrics_aggregated


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



