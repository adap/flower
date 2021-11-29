import flwr as fl
from numpy import save
import ray
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from flwr.common.typing import Scalar
from flwr.dataset.utils.common import create_lda_partitions, XYList
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from typing import Dict, Callable, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip

transforms_train = Compose(
    [
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transforms_test = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)


def get_resnet18_gn(num_classes: int = 10):
    model = resnet18(norm_layer=lambda x: nn.GroupNorm(2, x), num_classes=num_classes)
    return model


class FL_CIFAR(Dataset):
    def __init__(self, path_to_file: Path, transform=None):
        super().__init__()
        self.transform = transform
        self.X, self.Y = torch.load(path_to_file)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
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


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir: Path):
        self.cid = cid
        self.fed_dir = fed_dir
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net=None):
        if not net:
            net = get_resnet18_gn()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        net = get_resnet18_gn()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net

    def fit(self, parameters, config):
        print(f"fit() on client cid={self.cid}")
        net = self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])

        trainset = FL_CIFAR(
            path_to_file=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=transforms_train,
        )
        print("pre")
        # trainloader = DataLoader(
        #    trainset, batch_size=int(config["batch_size"]), workers=num_workers
        # )
        trainloader = DataLoader(trainset, batch_size=5)
        print("pos")
        print("hello3")
        # send model to device
        net.to(self.device)

        # train
        train(net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        validationset = FL_CIFAR(path_dir=Path(self.fed_dir) / self.cid / "test.pt")
        valloader = DataLoader(validationset, batch_size=50, workers=num_workers)

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(5),
        "batch_size": str(64),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_resnet18_gn()
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    # Generate partitions
    root_dir = "./data"
    fed_dir = Path(root_dir) / "partitions"
    trainset = CIFAR10(root=root_dir, train=True, download=True)
    XYData = (trainset.data, np.array(trainset.targets, dtype=np.long))
    train_partitions, dist = create_lda_partitions(
        dataset=XYData, num_partitions=500, concentration=0.1
    )
    for idx, partition in enumerate(train_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / "train.pt")

    # Prepare centralized test
    testset = CIFAR10(
        root="./data", train=False, download=True, transform=transforms_test
    )
    pool_size = 500  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 1}  # each client will get allocated 1 CPUs

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),
    )

    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(cid, fed_dir)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=5,
        strategy=strategy,
        ray_init_args=ray_config,
    )


if __name__ == "__main__":
    main()
