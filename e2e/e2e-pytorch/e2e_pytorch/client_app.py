import warnings
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import ConfigsRecord, Context, RecordSet

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SUBSET_SIZE = 1000
STATE_VAR = "timestamp"


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./../data", train=True, download=True, transform=trf)
    testset = CIFAR10("./../data", train=False, download=True, transform=trf)
    trainset = Subset(trainset, range(SUBSET_SIZE))
    testset = Subset(testset, range(10))
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(NumPyClient):

    def __init__(self, state: RecordSet):
        self.state = state

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def _record_timestamp_to_state(self):
        """Record timestamp to client's state."""
        t_stamp = datetime.now().timestamp()
        value = str(t_stamp)
        if STATE_VAR in self.state.configs_records.keys():
            value = self.state.configs_records[STATE_VAR][STATE_VAR]  # type: ignore
            value += f",{t_stamp}"

        self.state.configs_records[STATE_VAR] = ConfigsRecord({STATE_VAR: value})

    def _retrieve_timestamp_from_state(self):
        return self.state.configs_records[STATE_VAR][STATE_VAR]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        self._record_timestamp_to_state()
        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {STATE_VAR: self._retrieve_timestamp_from_state()},
        )

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        self._record_timestamp_to_state()
        return (
            loss,
            len(testloader.dataset),
            {"accuracy": accuracy, STATE_VAR: self._retrieve_timestamp_from_state()},
        )


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return


def client_fn(context: Context):
    return FlowerClient(context.state).to_client()


app = ClientApp(
    client_fn=client_fn,
)


if __name__ == "__main__":
    # Start Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(state=RecordSet()).to_client(),
    )
