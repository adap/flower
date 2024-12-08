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


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable


# normalize = transforms.Normalize(
#     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
# )
# transform_train = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Lambda(
#             lambda x: F.pad(
#                 Variable(x.unsqueeze(0), requires_grad=False),
#                 (4, 4, 4, 4),
#                 mode="reflect",
#             ).data.squeeze()
#         ),
#         transforms.ToPILImage(),
#         transforms.RandomCrop(32),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]
# )
normalize = transforms.Normalize(
    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
)

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ]
)

def apply_transforms(batch):
        # For CIFAR-10 the "img" column contains the images we want to
        # apply the transforms to
        batch["img"] = [transform_train(img) for img in batch["img"]]
        # map to a common column just to implify training loop
        # Note "label" doesn't exist in CIFAR-100
        batch["label"] = batch["fine_label"]
        return batch


def get_dataloader(num_partitions:int, beta: float, dataset_name:str, batch_size: 32, partition_id: int):

    def _get_dataloader(partition):

        partition.set_format("torch")
        return DataLoader(
            partition.with_transform(apply_transforms),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        alpha=beta,
        partition_by="fine_label",
        seed=1234,
    )
    dataset = FederatedDataset(
        dataset="uoft-cs/cifar100",
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
