"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import copy
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moon.models import init_net, train_fedprox, train_moon
from moon.dataset import get_train_transforms, get_apply_transforms_fn


# pylint: disable=too-many-instance-attributes
class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        # net: torch.nn.Module,
        net_id: int,
        dataset: str,
        model: str,
        output_dim: int,
        trainloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        mu: float,
        temperature: float,
        model_dir: str,
        alg: str,
    ):  # pylint: disable=too-many-arguments
        self.net = init_net(dataset, model, output_dim)
        self.net_id = net_id
        self.dataset = dataset
        self.model = model
        self.output_dim = output_dim
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mu = mu  # pylint: disable=invalid-name
        self.temperature = temperature
        self.model_dir = model_dir
        self.alg = alg

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        prev_net = init_net(self.dataset, self.model, self.output_dim)
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            prev_net = copy.deepcopy(self.net)
        else:
            # load previous model from model_dir
            prev_net.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, str(self.net_id), "prev_net.pt")
                )
            )
        global_net = init_net(self.dataset, self.model, self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        if self.alg == "moon":
            train_moon(
                self.net,
                global_net,
                prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device,
            )
        elif self.alg == "fedprox":
            train_fedprox(
                self.net,
                global_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.device,
            )
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(
            self.net.state_dict(),
            os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
        )
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}


def get_dataloader(num_partitions:int, beta: float, dataset_name:str, batch_size: 32, partition_id: int):

    def _get_dataloader(partition):

        partition.set_format("torch")
        tt = get_train_transforms(dataset_name)

        return DataLoader(
            partition.with_transform(get_apply_transforms_fn(tt, dataset_name)),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        alpha=beta,
        partition_by="fine_label" if dataset_name == "uoft-cs/cifar100" else "label",
        seed=1234,
    )
    dataset = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )
    return _get_dataloader(dataset.load_partition(partition_id = partition_id))


def gen_client_fn(
    cfg: DictConfig,
) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trainloader = get_dataloader(cfg.num_clients,
                                    cfg.dataset.beta,
                                    cfg.dataset.name,
                                    cfg.batch_size,
                                    int(cid))

        return FlowerClient(
            int(cid),
            cfg.dataset.name,
            cfg.model.name,
            cfg.model.output_dim,
            trainloader,
            device,
            cfg.num_epochs,
            cfg.learning_rate,
            cfg.mu,
            cfg.temperature,
            cfg.model.dir,
            cfg.alg,
        )

    return client_fn
