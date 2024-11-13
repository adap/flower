"""fedrs: A Flower Baseline."""

from collections import OrderedDict

import torch
import torchvision
from torch import nn

VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

# VGG code adapted from
# https://github.com/adap/flower/blob/main/baselines/fednova/fednova/models.py
class VGG(nn.Module):
    """Define 9-layer VGG model."""

    def __init__(self, num_classes: int, use_bn: bool):
        super().__init__()
        self.features = make_layers(VGG11_CFG, use_bn)
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(network_cfg, batch_norm=False):
    """Define the layer configuration of the VGG-16 network."""
    layers = []
    in_channels = 3
    for v in network_cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def initialize_model(name, num_classes):
    """Initialize the model with the given name."""
    model_functions = {
        "resnet18": lambda: torchvision.models.resnet18(),
        "vgg11": lambda: VGG(num_classes, use_bn=False),
    }
    model = model_functions[name]()

    if name.startswith("resnet"):
        num_ftrs = model.fc.in_features
        # Modify final layer to match the number of classes
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def train(
    net,
    trainloader,
    local_epochs,
    device,
    learning_rate,
    criterion,
    momentum,
    weight_decay,
):
    """Train the model on the training set."""
    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    running_loss = 0.0
    for _ in range(local_epochs):
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
    correct, loss = 0.0, 0.0
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
