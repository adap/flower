"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from typing import List, Tuple

import torch
from flwr.common import Metrics
from omegaconf import DictConfig
import matplotlib.pyplot as plt


def comp_accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
	# Multiply accuracy of each client by number of examples used
	accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
	examples = [num_examples for num_examples, _ in metrics]

	# Aggregate and return custom metric (weighted average)
	return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(exp_config: DictConfig, server_round: int):
	"""Return training configuration dict for each round.
	Learning rate is reduced by a factor after set rounds.
	"""

	config = {}

	lr = exp_config.optimizer.lr

	if exp_config.lr_scheduling:
		if server_round == int(exp_config.num_rounds / 2):
			lr = exp_config.optimizer.lr / 10

		elif server_round == int(exp_config.num_rounds * 0.75):
			lr = exp_config.optimizer.lr / 100

	config["lr"] = lr
	config["server_round"] = server_round
	return config


def generate_plots():
	"""Generate plots for the experiment."""
	metrics = ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]
	for metric in metrics[-1]:
		pass



if __name__ == "__main__":
	pass
