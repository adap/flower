from collections import OrderedDict

import flwr as fl
import torch
import nvsmi
import uuid

from argparse import ArgumentParser
from copy import deepcopy
from socket import getfqdn
from utils import valid_folder
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from flwr.common.typing import Config, NDArrays, Scalar
from datasets import OpenImage
from pathlib import Path


from torchvision.models import shufflenet_v2_x2_0
from flwr.server.strategy.aggregate import aggregate


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)

Net = shufflenet_v2_x2_0

# Define Flower client
class FlowerWorker(fl.client.NumPyClient):
    def __init__(
        self, *, worker_id: Optional[str] = None, cid_csv_root: Path, img_root: Path
    ):
        self.worker_id = worker_id if worker_id else uuid.uuid4().hex
        self.node_name = getfqdn()
        self.net = Net(num_classes=596)
        self.partially_agg_model: Optional[Tuple[NDArrays, int]]
        self.dataset = OpenImage(cid_csv_root=cid_csv_root, img_root=img_root)

    def partially_aggregate(self, new_result: Tuple[NDArrays, int]) -> None:
        if self.partially_agg_model is not None:
            total_samples = self.partially_agg_model[1] + new_result[1]
            temp = aggregate([self.partially_agg_model, new_result])
            self.partially_agg_model = (temp, total_samples)
        else:
            self.partially_agg_model = new_result

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        node: Dict[str, Scalar] = {"worker_id": self.worker_id}
        node["node_name"] = self.node_name
        for gpu in nvsmi.get_gpus():
            node[gpu.uuid] = node[gpu.to_json()]

        return node

    def train_single_client(
        self,
        *,
        trainloader: DataLoader,
        device: torch.device,
        lr: float,
        momentum: float,
        weight_decay: float,
        epochs: int,
    ):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        for _ in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                criterion(self.net(images.to(device)), labels.to(device)).backward()
                optimizer.step()

    def test_single_client(
        self, *, testloader: DataLoader, device: torch.device
    ) -> Tuple[float, float]:
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = self.net(images.to(device))
                labels = labels.to(device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        return loss / total, correct / total

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        device = torch.device(f"cuda:{str(config['gpu_id'])}")
        list_clients = str(config["list_clients"]).split("_")
        for actual_cid in list_clients:
            # Make copy of parameters for new client
            this_client_parameters = deepcopy(parameters)
            self.set_parameters(this_client_parameters)

            # Prepare dataset for new client
            self.dataset.load_client(cid=actual_cid, dataset_type="train")
            train_dataloader = DataLoader(
                self.dataset,
                batch_size=int(config["batch_size"]),
                pin_memory=True,
                num_workers=2,
            )

            # Train actual clients
            lr, momentum, wd, epochs = (
                float(config["lr"]),
                float(config["momentum"]),
                float(config["wd"]),
                int(config["epochs"]),
            )
            self.train_single_client(
                trainloader=train_dataloader,
                device=device,
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
                epochs=epochs,
            )
            locally_trained_model = self.get_parameters(config={})
            self.partially_aggregate((locally_trained_model, len(self.dataset)))

        return self.partially_agg_model, {}

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ):
        device = torch.device(f"cuda:{str(config['gpu_id'])}")
        list_clients = str(config["list_clients"]).split("_")
        total_correct, total_loss, total_num_samples = 0.0, 0.0, 0
        for actual_cid in list_clients:
            # Make copy of parameters for new client
            this_client_parameters = deepcopy(parameters)
            self.set_parameters(this_client_parameters)

            # Prepare dataset for new client
            self.dataset.load_client(cid=actual_cid, dataset_type="test")
            num_samples = len(self.dataset)
            test_dataloader = DataLoader(
                self.dataset,
                batch_size=int(config["batch_size"]),
                pin_memory=True,
                num_workers=2,
            )
            loss, correct = self.test_single_client(
                testloader=test_dataloader,
                device=device,
            )
            total_correct += correct
            total_loss += loss
            num_samples = total_num_samples + num_samples
        acc = total_correct / total_num_samples
        return acc, total_num_samples, {"accuracy": acc}


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Flower Worker Wrapper",
        description="This is a pseudo-client that runs multiple virtual clients in a single process.",
    )
    parser.add_argument(
        "--path_imgs",
        type=str,
        default="/datasets/FedScale/openImg/",
        help="Root directory containing 'train' and 'test' folders with images.",
    )
    parser.add_argument(
        "--path_csv_map",
        type=valid_folder,
        default="/datasets/FedScale/openImg/client_data_mapping/clean_ids/",
        help="Root directory containing 'train' and 'test' folders with {virtual_client_id}.csv mapping files.",
    )
    args = parser.parse_args()
    # Start Flower Worker
    img_root = Path("/datasets/FedScale/openImg/")
    cid_csv_root = Path("/datasets/FedScale/openImg/client_data_mapping/clean_ids")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerWorker(cid_csv_root=cid_csv_root, img_root=img_root),
    )
