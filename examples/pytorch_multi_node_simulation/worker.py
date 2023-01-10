from collections import OrderedDict
import time

import flwr as fl
import torch
import nvsmi
import uuid

from argparse import ArgumentParser
from copy import deepcopy
from socket import getfqdn
from utils import valid_folder
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from flwr.common.typing import Config, NDArrays, Scalar
from datasets import OpenImage
from pathlib import Path


from torchvision.models import shufflenet_v2_x2_0
from utils import aggregate_pytorch_tensor


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
        self.num_classes = 596
        self.worker_id = worker_id if worker_id else uuid.uuid4().hex
        self.node_name = getfqdn()
        self.net = Net(num_classes=self.num_classes)
        self.gpu_partially_aggregated_model: Tuple[List[torch.Tensor], int] = ([], 0)
        self.dataset = OpenImage(cid_csv_root=cid_csv_root, img_root=img_root)
        self.fake_data = (
            torch.rand((20, 3, 254, 254), dtype=torch.float, device="cuda:0"),
            torch.randint(low=0, high=self.num_classes, size=(20,), device="cuda:0"),
        )

    def partially_aggregate(self, new_result: Tuple[List[torch.Tensor], int]) -> None:
        if self.gpu_partially_aggregated_model[0]:  # Not empty
            total_samples = self.gpu_partially_aggregated_model[1] + new_result[1]
            temp = aggregate_pytorch_tensor(
                [self.gpu_partially_aggregated_model, new_result]
            )
            self.gpu_partially_aggregated_model = (temp, total_samples)
        else:
            self.gpu_partially_aggregated_model = new_result

    def zero_partially_aggregated_model(self):
        if self.gpu_partially_aggregated_model is not None:
            agg_model, _ = self.gpu_partially_aggregated_model
            for t in agg_model:
                t.zero_()
            self.gpu_partially_aggregated_model = (agg_model, 0)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        node: Dict[str, Scalar] = {"worker_id": self.worker_id}
        node["node_name"] = self.node_name
        for gpu in nvsmi.get_gpus():
            node[gpu.uuid] = gpu.to_json()

        return node

    def train_single_client(
        self,
        *,
        net,
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
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        for _ in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                criterion(self.net(images.to(device)), labels.to(device)).backward()
                optimizer.step()

    def test_single_client(
        self, *, net, testloader: DataLoader, device: torch.device
    ) -> Tuple[float, float]:
        """Validate the model on the test set."""
        net.eval()
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = net(images.to(device))
                labels = labels.to(device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        return loss / total, correct / total

    def get_parameters(self, config, net=None, device="cpu"):
        if net is None:
            net = self.net
        if device == "cpu":
            tmp = [
                val.detach().to(device).numpy() for _, val in net.state_dict().items()
            ]
        else:
            tmp = [val.detach().to(device) for _, val in net.state_dict().items()]
        return tmp

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        list_clients = str(config["list_clients"]).split("_")
        gpu_id = str(config["gpu_id"])
        device = torch.device(f"cuda:{gpu_id}")
        self.set_parameters(parameters)
        self.net.to(device)

        for virtual_cid in list_clients:
            # Make copy of parameters for new client
            cloned_net = deepcopy(self.net)
            cloned_net.train()
            # Prepare dataset for new client
            self.dataset.load_client(cid=virtual_cid, dataset_type="train")
            train_dataloader = DataLoader(
                self.dataset,
                batch_size=int(config["batch_size"]),
                pin_memory=True,
                num_workers=4,
            )

            # Train actual clients
            lr, momentum, wd, epochs = (
                float(config["lr"]),
                float(config["momentum"]),
                float(config["weight_decay"]),
                int(config["epochs"]),
            )
            self.train_single_client(
                net=cloned_net,
                trainloader=train_dataloader,
                device=device,
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
                epochs=epochs,
            )
            # Accumulate partial result in GPU
            cloned_net.eval()
            locally_trained_model = self.get_parameters(
                net=cloned_net, config={}, device=f"cuda:{gpu_id}"
            )
            self.partially_aggregate((locally_trained_model, len(self.dataset)))

        # Convert model back to CPU to send it to Aggregation Server
        cpu_partially_aggregated_model: NDArrays = [
            x.cpu().numpy() for x in self.gpu_partially_aggregated_model[0]
        ]
        total_num_samples = self.gpu_partially_aggregated_model[1]

        # Clear partially aggregated model.
        self.zero_partially_aggregated_model()
        return cpu_partially_aggregated_model, total_num_samples, {}
        # return self.partially_agg_model[0], self.partially_agg_model[1], {}

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
                net=self.net,
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
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerWorker(
            cid_csv_root=Path(args.path_csv_map), img_root=Path(args.path_imgs)
        ),
    )
