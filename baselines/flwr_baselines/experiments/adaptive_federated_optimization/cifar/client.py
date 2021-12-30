from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from .utils import get_model, train, test

transforms_test = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transforms_train = Compose([RandomHorizontalFlip(), transforms_test])


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
    def __init__(self, cid: str, fed_dir: Path):
        self.cid = cid
        self.fed_dir = fed_dir
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
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
        net = self.set_parameters(parameters)
        net.to(self.device)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = ClientDataset(
            path_to_data=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=transforms_train,
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
            path_to_data=Path(self.fed_dir) / self.cid / "test.pt"
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
