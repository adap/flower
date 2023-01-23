from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import flwr as fl
from opacus import PrivacyEngine

# Adapted from the PyTorch quickstart example.


# Define parameters.
PARAMS = {
    "batch_size": 32,
    "train_split": 0.7,
    "local_epochs": 1,
}
PRIVACY_PARAMS = {
    # 'target_epsilon': 5.0,
    "target_delta": 1e-5,
    "noise_multiplier": 0.4,
    "max_grad_norm": 1.2,
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define model used for training.
class Net(nn.Module):
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
        x = self.fc3(x)
        return x


def train(net, trainloader, privacy_engine, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    privacy_engine.attach(optimizer)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    epsilon, _ = optimizer.privacy_engine.get_privacy_spent(
        PRIVACY_PARAMS["target_delta"]
    )
    return epsilon


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# Define Flower client.
class DPCifarClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, sample_rate) -> None:
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        # Create a privacy engine which will add DP and keep track of the privacy budget.
        self.privacy_engine = PrivacyEngine(
            self.model,
            sample_rate=sample_rate,
            target_delta=PRIVACY_PARAMS["target_delta"],
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epsilon = train(
            self.model, self.trainloader, self.privacy_engine, PARAMS["local_epochs"]
        )
        print(f"epsilon = {epsilon:.2f}")
        return self.get_parameters(config={}), len(self.trainloader), {"epsilon": epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
