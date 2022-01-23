from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from flwr.common.parameter import weights_to_parameters
from flwr.common.typing import Parameters, Scalar, Weights
from flwr.dataset.utils.common import XY, create_lda_partitions
from flwr.server.history import History
from PIL import Image
from torch import Tensor, load, save
from torch.nn import GroupNorm, Module
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet, resnet18
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

# transforms
def get_transforms(num_classes: int = 10) -> Dict[str, Compose]:
    normalize_cifar10=Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize_cifar100=Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    normalize_cifar = normalize_cifar10 if num_classes == 10 else normalize_cifar100
    train_transform = Compose([
        RandomCrop(24),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize_cifar
    ])
    test_transform = Compose([
        CenterCrop(24),
        ToTensor(),
        normalize_cifar
    ])
    return {"train": train_transform, "test": test_transform}


def get_cifar_model(num_classes: int = 10) -> Module:
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model


class ClientDataset(Dataset):
    def __init__(self, path_to_data: Path, transform: Compose = None):
        super().__init__()
        self.transform = transform
        self.X, self.Y = load(path_to_data)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def partition_cifar10_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: Optional[npt.NDArray[np.float32]] = None,
    num_partitions: int = 500,
    concentration: float = 0.1,
) -> np.ndarray:
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
    )
    # Save partions
    for idx, partition in enumerate(clients_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / "train.pt")

    return dist

def partition_cifar100_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: Optional[npt.NDArray[np.float32]] = None,
    num_partitions: int = 500,
    concentration_coarse: float = 0.1,
    concentration_fine: float = 0.1,
) -> np.ndarray:
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration_coarse,
    )
    # Save partions
    for idx, partition in enumerate(clients_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / "train.pt")

    return dist


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    def on_fit_config(rnd: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": rnd,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def get_cifar_eval_fn(
    path_original_dataset: Path, num_classes: int = 10
) -> Callable[[Weights], Optional[Tuple[float, Dict[str, float]]]]:
    """Returns an evaluation function for centralized evaluation."""
    CIFAR = CIFAR10 if num_classes == 10 else CIFAR100
    transforms = get_transforms(num_classes=num_classes)

    testset = CIFAR(
        root=path_original_dataset,
        train=False,
        download=True,
        transform=transforms["test"],
    )

    def evaluate(weights: Weights) -> Optional[Tuple[float, Dict[str, float]]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_cifar_model(num_classes=num_classes)
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
) -> Path:
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration:.2f}"
    )

    trainset = CIFAR10(root=path_original_dataset, train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int32))
    partition_cifar10_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
    )

    return fed_dir

def gen_cifar100_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
    lda_concentration_coarse: float,
    lda_concentration_fine: float,
) -> Path:
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration_coarse:.2f}_{lda_concentration_fine:.2f}"
    )

    trainset = CIFAR100(root=path_original_dataset, train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int32))
    partition_cifar100_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration_coarse=lda_concentration_coarse,
        concentration_fine=lda_concentration_fine,
    )

    return fed_dir

def get_initial_parameters(num_classes: int = 10) -> Parameters:
    model = get_cifar_model(num_classes)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = weights_to_parameters(weights)

    return parameters


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    metric_str: str,
    strategy_name: str,
    expected_maximum: float,
    save_path: Path,
) -> None:
    x, y = zip(*hist.metrics_centralized[metric_str])
    plt.figure()
    plt.plot(x, np.asarray(y) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.close()
