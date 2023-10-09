"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from typing import List, Tuple

import torch
from flwr.common import Metrics
from omegaconf import DictConfig

class Meter(object):
	""" Computes and stores the average, variance, and current value """

	def __init__(self, init_dict=None, ptag='Time', stateful=False,
				 csv_format=True):
		"""
		:param init_dict: Dictionary to initialize meter values
		:param ptag: Print tag used in __str__() to identify meter
		:param stateful: Whether to store value history and compute MAD
		"""
		self.reset()
		self.ptag = ptag
		self.value_history = None
		self.stateful = stateful
		if self.stateful:
			self.value_history = []
		self.csv_format = csv_format
		if init_dict is not None:
			for key in init_dict:
				try:
					self.__dict__[key] = init_dict[key]
				except Exception:
					print('(Warning) Invalid key {} in init_dict'.format(key))

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.std = 0
		self.sqsum = 0
		self.mad = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		self.sqsum += (val ** 2) * n
		if self.count > 1:
			self.std = ((self.sqsum - (self.sum ** 2) / self.count)
						/ (self.count - 1)
						) ** 0.5
		if self.stateful:
			self.value_history.append(val)
			mad = 0
			for v in self.value_history:
				mad += abs(v - self.avg)
			self.mad = mad / len(self.value_history)

	def __str__(self):
		if self.csv_format:
			if self.stateful:
				return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
						   .format(dm=self))
			else:
				return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
						   .format(dm=self))
		else:
			if self.stateful:
				return str(self.ptag) + \
					   str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
						   .format(dm=self))
			else:
				return str(self.ptag) + \
					   str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
						   .format(dm=self))


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
