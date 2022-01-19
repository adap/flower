from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import flwr as fl
import numpy as np
import ray
import torch
from cifar.utils import ClientDataset, get_cifar_model, get_transforms, test, train
from flwr.common import parameter, weights_to_parameters
from flwr.common.parameter import Parameters
from flwr.common.typing import Scalar, Weights
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class RayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir: Path, num_classes: int):
        self.cid = cid
        self.fed_dir = fed_dir
        self.num_classes = num_classes
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net=None) -> Weights:
        if net is None:
            net = get_cifar_model(self.num_classes)
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        # parameters = weights_to_parameters(weights)
        return weights

    def get_properties(self, ins: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return self.properties

    def fit(
        self, parameters: Weights, config: Dict[str, Scalar]
    ) -> Tuple[Weights, int, Dict[str, Scalar]]:
        net = self.set_parameters(parameters)
        net.to(self.device)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = ClientDataset(
            path_to_data=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=get_transforms(self.num_classes)["train"],
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
            transform=get_transforms()["test"],
        )
        valloader = DataLoader(validationset, batch_size=50, num_workers=num_workers)

        # send model to device
        net.to(self.device)

        # evaluate
        loss, accuracy = test(net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def set_parameters(self, parameters: Weights):
        net = get_cifar_model(self.num_classes)
        weights = parameters
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net


def get_ray_client_fn(
    fed_dir: Path, num_classes: int = 10
) -> Callable[[str], RayClient]:
    def client_fn(cid: str) -> RayClient:
        # create a single client instance
        return RayClient(cid=cid, fed_dir=fed_dir, num_classes=num_classes)

    return client_fn
