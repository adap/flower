from typing import Callable, Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from fednova.models import test, train


class FlowerClient(fl.client.NumPyClient):  # pylint: disable=too-many-instance-attributes
	"""Standard Flower client for CNN training."""

	def __init__(self, net: torch.nn.Module, client_id: str, trainloader: DataLoader,
				 valloader: DataLoader, device: torch.device, num_epochs: int, ratio: float, config: DictConfig):

		self.net = net
		self.exp_config = config
		self.optimizer = instantiate(config.optimizer, params=self.net.parameters(), ratio=ratio)
		self.trainLoader = trainloader
		self.valLoader = valloader
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

		self.optimizer.set_model_params(parameters)

	def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
		"""Implements distributed fit function for a given client."""

		self.set_parameters(parameters)
		self.optimizer.set_lr(config["lr"])

		if self.exp_config.var_local_epochs:
			seed_val = 2023 + int(self.client_id) + config["server_round"] + self.exp_config.seed
			np.random.seed(seed_val)
			num_epochs = np.random.randint(self.exp_config.var_min_epochs, self.exp_config.var_max_epochs)
		else:
			num_epochs = self.num_epochs

		train(self.net,
			  self.optimizer,
			  self.trainLoader,
			  self.device,
			  epochs=num_epochs)

		# Get ratio by which the strategy would scale the local gradients from each client
		# We use this scaling factor to aggregate the gradients on the server
		grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()

		return self.get_parameters({}), len(self.trainLoader), grad_scaling_factor

	def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
		"""Implements distributed evaluation for a given client."""

		# Evaluation ideally is done on validation set, but because we already know the best hyper-parameters from the paper
		# and since individual client datasets are already quite small, we merge the validation set with the training set
		# and evaluate on the training set with the aggregated global model parameters.
		# This behaviour can be modified by passing the validation set in the below test(self.valLoader) function
		# and replacing len(self.valLoader) below.
		# Note that we evaluate on the centralized test-set on server-side in the strategy.

		self.set_parameters(parameters)
		loss, accuracy = test(self.net, self.trainLoader, self.device)
		return float(loss), len(self.trainLoader), {"accuracy": float(accuracy)}


def gen_client_fn(num_epochs: int, trainloaders: List[DataLoader], testloader: DataLoader, data_ratios: List,
				model: DictConfig, exp_config: DictConfig) -> Callable[[str], FlowerClient]:

	def client_fn(cid: str) -> FlowerClient:
		"""Create a Flower client representing a single organization."""

		# Load model
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		net = instantiate(model)

		# Note: each client gets a different trainloader/valloader, so each client
		# will train and evaluate on their own unique data
		trainloader = trainloaders[int(cid)]
		client_dataset_ratio = data_ratios[int(cid)]

		return FlowerClient(
			net,
			cid,
			trainloader,
			testloader,
			device,
			num_epochs,
			client_dataset_ratio,
			exp_config
		)

	return client_fn
