from typing import Tuple, List, Dict
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr.common import Metrics, NDArrays
from flwr.common.typing import Scalar
from datasets.utils.logging import disable_progress_bar

from .model import NetResnet18
from evaluation.eval_utils import apply_transforms_test, test


# transformation to convert images to tensors and apply normalization
def apply_transforms_train(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def set_params(model, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Train loop
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct_sum, loss_sum = 0, 0.0
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optim.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            loss_sum += loss.detach()
            _, predicted = torch.max(outputs.data, 1)
            correct_sum += (predicted == labels).sum().item()

    accuracy = correct_sum / len(trainloader.dataset)
    return loss_sum, accuracy


def get_evaluate_fn(centralized_testset, save_every_round, save_path):
    """Return an evaluation function for saving global model."""
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ):
        # Init model
        model = NetResnet18()
        set_params(model, parameters)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms_test)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # Save model
        if server_round % save_every_round == 0:
            torch.save(model.state_dict(), f"{save_path}/aggregated_model_{server_round}.pth")

        return loss, {"accuracy": accuracy}
    return evaluate


def get_fit_config(local_epoch, batch_size, lr, momentum):
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": local_epoch,  # Number of local epochs done by clients
            "batch_size": batch_size,  # Batch size to use by clients during fit()
            "lr": lr,  # Learning rate
            "momentum": momentum,  # Momentum value for optimiser
        }
        return config
    return fit_config


def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics"""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    accuracies = [num_examples * m["train_acc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples), "train_accuracy": sum(accuracies) / sum(examples)}


def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
