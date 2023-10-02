from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import os
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models import test, train


class FlowerClient(fl.client.NumPyClient):  # pylint: disable=too-many-instance-attributes
	"""Standard Flower client for CNN training."""

	def __init__(self, net: torch.nn.Module, client_state: Dict, client_id: str, trainloader: DataLoader,
				 testloader: DataLoader,device: torch.device, num_epochs: int, ratio: float, config: DictConfig):

		self.net = net
		self.exp_config = config
		self.optimizer = instantiate(config.optimizer, params=self.net.parameters(), ratio=ratio)
		self.trainloader = trainloader
		self.optim_state = client_state
		# self.valloader = valloader
		self.testloader = testloader
		self.client_id = client_id
		self.device = device
		self.num_epochs = num_epochs
		self.data_ratio = ratio

	def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
		"""Returns the parameters of the current net."""
		params = [val["cum_grad"].cpu().numpy() for _, val in self.optimizer.state_dict()["state"].items()]
		return params

	def set_parameters(self, parameters: NDArrays) -> None:
		"""Changes the parameters of the model using the given ones."""
		if self.optim_state is not None:
			self.optimizer.load_state_dict(self.optim_state)

		self.optimizer.load_aggregated_grad(parameters)

	def fit(self, parameters: NDArrays, config: DictConfig) -> Tuple[NDArrays, int, Dict]:
		"""Implements distributed fit function for a given client."""
		self.set_parameters(parameters)

		# Adjust learning rate based on server rounds
		current_round = config["server_round"]

		if current_round == int(self.exp_config.num_rounds / 2):
			lr = self.exp_config.optimizer.lr / 10
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

		elif current_round == int(self.exp_config.num_rounds * 0.75):
			lr = self.exp_config.optimizer.lr / 100
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

		if self.exp_config.var_local_epochs:
			num_epochs = np.random.randint(self.exp_config.var_min_epochs, self.exp_config.var_max_epochs)
		else:
			num_epochs = self.num_epochs

		train(self.net,
			  self.optimizer,
			  self.trainloader,
			  self.device,
			  epochs=num_epochs)

		local_stats = self.optimizer.get_local_stats()

		self.optim_state = self.optimizer.state_dict()
		torch.save(self.optim_state, f"state/client_{self.client_id}_state.pt")

		return self.get_parameters({}), len(self.trainloader), local_stats

	def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
		"""Implements distributed evaluation for a given client."""
		self.set_parameters(parameters)
		loss, accuracy = test(self.net, self.testloader, self.device)
		# print("-----Client: {} Round: {} Test Loss: {} Accuracy : {} ".format(self.client_id, config["server_round"],
		# 																	  loss, accuracy))
		return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


def gen_client_fn(num_epochs: int, trainloaders: List[DataLoader], testloader: DataLoader, data_ratios: List,
				model: DictConfig, exp_config: DictConfig) -> Callable[[str], FlowerClient]:

	def client_fn(cid: str) -> FlowerClient:
		"""Create a Flower client representing a single organization."""

		# Load model
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		net = instantiate(model).to(device)
		# net = vgg11().to(device)

		if os.path.exists(f"state/client_{cid}_state.pt"):
			client_state = torch.load(f"state/client_{cid}_state.pt", map_location=device)
		else:
			client_state = None

		# Note: each client gets a different trainloader/valloader, so each client
		# will train and evaluate on their own unique data
		trainloader = trainloaders[int(cid)]
		client_dataset_ratio = data_ratios[int(cid)]
		# valloader = valloaders[int(cid)]

		return FlowerClient(
			net,
			client_state,
			cid,
			trainloader,
			testloader,
			device,
			num_epochs,
			client_dataset_ratio,
			exp_config
		)

	return client_fn
