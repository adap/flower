import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr_datasets import FederatedDataset

from utils import train, set_params, apply_transforms_train
from model import Net
from evaluation.eval_utils import test


# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset

        # Instantiate model
        self.model = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]
        lr, momentum = config["lr"], config["momentum"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # Train
        loss, accuracy = train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        metric = {"train_loss": loss, "train_acc": accuracy}

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), metric

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        testloader = DataLoader(self.testset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}


def get_client_fn(dataset: FederatedDataset, test_size: float, seed: int):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), "train")

        # Client train/test split
        client_dataset_splits = client_dataset.train_test_split(test_size=test_size, seed=seed)
        trainset = client_dataset_splits["train"]
        testset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms_train)
        testset = testset.with_transform(apply_transforms_train)

        # Create and return client
        return FlowerClient(trainset, testset)

    return client_fn
