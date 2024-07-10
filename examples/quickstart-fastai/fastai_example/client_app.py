"""fastai_example: A Flower / Fastai app."""

import warnings
from collections import OrderedDict
from typing import Any

import torch

# from fastai.vision.all import *
from fastai.vision.all import (
    ImageDataLoaders,
    URLs,
    error_rate,
    squeezenet1_1,
    untar_data,
    vision_learner,
)
from fastai.vision.data import DataLoaders
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from flwr.client import Client, ClientApp, NumPyClient

warnings.filterwarnings("ignore", category=UserWarning)

# Download MNIST dataset
# path = untar_data(URLs.MNIST)

# # Load dataset
# dls = ImageDataLoaders.from_folder(
#     path, valid_pct=0.5, train="training", valid="testing", num_workers=0
# )
#
# # Define model
# learn = vision_learner(dls, squeezenet1_1, metrics=error_rate)


def load_data(partition_id) -> tuple[DataLoader, DataLoader, DataLoader]:
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    partition = fds.load_partition(partition_id, "train")

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            transforms.functional.to_tensor(img) for img in batch["image"]
        ]
        return batch

    def collate_fn(batch):
        """Change the dictionary to tuple to keep the exact dataloader behavior."""
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=1,
    )
    testloader = DataLoader(
        partition_full["test"], batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return trainloader, valloader, testloader


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, learn, dls) -> None:
        self.learn = learn
        self.dls = dls

    def get_parameters(self, config) -> list:
        return [val.cpu().numpy() for _, val in self.learn.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        params_dict = zip(self.learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.learn.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        self.set_parameters(parameters)
        self.learn.fit(1)
        return self.get_parameters(config={}), len(self.dls.train), {}

    def evaluate(self, parameters, config) -> tuple[Any, int, dict[str, Any]]:
        self.set_parameters(parameters)
        loss, error_rate = self.learn.validate()
        return loss, len(self.dls.valid), {"accuracy": 1 - error_rate}


def client_fn(node_id, partition_id) -> Client:
    """Client function to return an instance of Client()."""
    trainloader, valloader, _ = load_data(partition_id=partition_id)
    dls = DataLoaders(trainloader, valloader)
    learn = vision_learner(dls, squeezenet1_1, metrics=error_rate, n_out=10)
    return FlowerClient(learn, dls).to_client()


app = ClientApp(client_fn=client_fn)

# # Start Flower client
# fl.client.start_client(
#     server_address="127.0.0.1:8080",
#     client=FlowerClient().to_client(),
# )
