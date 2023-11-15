"""FedAvg and FedNB clients."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedbn.models import CNNModel, test, train


class FlowerClient(fl.client.NumPyClient):
    """"""

    def __init__(
        self,
        model: CNNModel,
        trainloader: DataLoader,
        testloader: DataLoader,
        dataset_name: str,
    ) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.dataset_name = dataset_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN
        layers.
        """
        # self.model.train() # TODO: is this needed ? check
        # Return all model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if
        available.
        """
        # self.model.train() # TODO: is this needed ? check
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Set model parameters, train model, return updated model parameters."""
        self.set_parameters(parameters)

        # train model on local dataset
        loss, acc = train(
            self.model,
            self.trainloader,
            epochs=1,
            device=self.device,
        )

        # construct metrics to return to server
        round = config["round"]
        metrics = {
            "dataset_name": self.dataset_name,
            "round": round,
            "train_acc": acc,
            "train_loss": loss,
            "num_train_examples": len(self.trainloader.dataset),
        }

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Set model parameters, evaluate model on local test dataset, return result."""
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return (
            float(loss),
            len(self.testloader.dataset),
            {"loss": loss, "accuracy": accuracy, "dataset_name": self.dataset_name},
        )


class FedBNFlowerClient(FlowerClient):
    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN
        layers.
        """
        # self.model.train() # TODO: is this needed ? check
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if
        available.
        """
        # self.model.train() # TODO: is this needed ? check
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)


def gen_client_fn(
    client_data: List[Tuple[DataLoader,DataLoader,int]],
    client_cfg: DictConfig,
    model_cfg: DictConfig,
) -> Callable[[str], FlowerClient]:
    """"""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Instantiate model
        net = instantiate(model_cfg)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader, valloader, dataset_name = client_data[int(cid)]
        return instantiate(
            client_cfg, model=net, trainloader=trainloader, testloader=valloader, dataset_name=dataset_name,
        )

    return client_fn
