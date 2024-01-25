from typing import Tuple, List
from collections import OrderedDict
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr.common import Metrics


# transformation to convert images to tensors and apply normalization
def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def set_params(model, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# borrowed from Pytorch quickstart example
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct_sum, loss_sum = 0, 0.0
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
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


def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    accuracies = [num_examples * m["train_acc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples), "train_accuracy": sum(accuracies) / sum(examples)}
