from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import flwr as fl
import numpy as np
import ray
import torch
from flwr.common.parameter import Parameters, Weights
from flwr.common.typing import Scalar
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from cifar.utils import (
    get_model,
    partition_and_save,
    train,
    test,
)
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor

# transforms
transform_cifar10_test = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transform_cifar100_test = Compose(
    [ToTensor(), Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
)
transform_cifar10_train = Compose([RandomHorizontalFlip(), transform_cifar10_test])
transform_cifar100_train = Compose([RandomHorizontalFlip(), transform_cifar100_test])


class ClientDataset(Dataset):
    def __init__(self, path_to_data: Path, transform: Compose = None):
        super().__init__()
        self.transform = transform
        self.X, self.Y = torch.load(path_to_data)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


class RayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir: Path, transforms: Dict[str, Compose]):
        self.cid = cid
        self.fed_dir = fed_dir
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.transforms = transforms
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net: Optional[Module] = None) -> Weights:
        if not net:
            net = get_model(Parameters)
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights

    def get_properties(self, ins: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return self.properties

    def fit(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        print(config)
        net = self.set_parameters(parameters)
        net.to(self.device)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = ClientDataset(
            path_to_data=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=self.transforms["train"],
        )

        trainloader = DataLoader(
            trainset, batch_size=int(config["batch_size"]), num_workers=num_workers
        )
        # train
        train(net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(net), len(trainloader.dataset), {}

    def evaluate(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, float]]:
        net = self.set_parameters(parameters)
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        validationset = ClientDataset(
            path_to_data=Path(self.fed_dir) / self.cid / "test.pt",
            transform=self.transforms["test"],
        )
        valloader = DataLoader(validationset, batch_size=50, num_workers=num_workers)

        # send model to device
        net.to(self.device)

        # evaluate
        loss, accuracy = test(net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def set_parameters(self, parameters):
        net = get_model()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net


def get_ray_client_fn(fed_dir: Path, dataset: str) -> Callable[[str], RayClient]:
    if dataset == "cifar10":
        transforms = {
            "train": transform_cifar10_train,
            "test": transform_cifar10_test,
        }

    def client_fn(cid: str) -> RayClient:
        # create a single client instance
        return RayClient(cid=cid, fed_dir=fed_dir, transforms=transforms)

    return client_fn


def get_cifar_eval_fn(
    path_original_dataset: Path, num_classes: int = 10
) -> Callable[[Weights], Optional[Tuple[float, Dict[str, float]]]]:
    """Returns an evaluation function for centralized evaluation."""
    CIFAR = CIFAR10 if num_classes == 10 else CIFAR100
    transform_test = (
        transform_cifar10_test if num_classes == 10 else transform_cifar100_test
    )

    testset = CIFAR(
        root=path_original_dataset,
        train=False,
        download=True,
        transform=transform_test,
    )

    def evaluate(weights: Weights) -> Optional[Tuple[float, Dict[str, float]]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_model()
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), weights)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(net, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def gen_cifar10_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
    lda_concentration: float,
) -> None:
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration:.2f}"
    )

    trainset = CIFAR10(root=path_original_dataset, train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int32))
    partition_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
    )

    return fed_dir


def get_initial_parameters(num_classes: int = 10) -> Parameters:
    model = get_cifar_model(num_classes)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = weights_to_parameters(weights)

    return parameters


def get_cifar_model(num_classes: int = 10) -> Module:
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model
