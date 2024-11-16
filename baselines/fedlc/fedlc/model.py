"""fedlc: A Flower Baseline."""

from collections import OrderedDict

import torch
import torchvision


# Adapted from FedDebug baseline implementation
# https://github.com/adap/flower/blob/main/baselines/feddebug/feddebug/models.py
def initialize_model(name, num_channels, num_classes):
    """Initialize the model with the given name."""
    model_functions = {
        "resnet18": lambda: torchvision.models.resnet18(),
        "resnet34": lambda: torchvision.models.resnet34(),
        "resnet50": lambda: torchvision.models.resnet50(),
        "resnet101": lambda: torchvision.models.resnet101(),
        "resnet152": lambda: torchvision.models.resnet152(),
        "vgg16": lambda: torchvision.models.vgg16(),
    }
    model = model_functions[name]()
    # Modify model for grayscale input if necessary
    if num_channels == 1:
        if name.startswith("resnet"):
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif name == "vgg16":
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )

    # Modify final layer to match the number of classes
    if name.startswith("resnet"):
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif name == "vgg16":
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
    return model


def train(net, trainloader, epochs, device, learning_rate, criterion):
    """Train the model on the training set."""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_parameters(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# implementation from DASHA paper
