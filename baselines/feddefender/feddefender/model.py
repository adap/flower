"""feddefender: A Flower Baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class ConvNet(nn.Module):
    """ConvNet model."""

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(config["channels"], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config["num_classes"])

    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _get_inputs_labels_from_batch(batch):
    if "image" in batch:
        return batch["image"], batch["label"]
    x, y = batch
    return x, y


def initialize_model(name, cfg):
    """Initialize the model with the given name."""
    model_functions = {
        "resnet18": lambda: torchvision.models.resnet18(weights="IMAGENET1K_V1"),
        "resnet34": lambda: torchvision.models.resnet34(weights="IMAGENET1K_V1"),
        "resnet50": lambda: torchvision.models.resnet50(weights="IMAGENET1K_V1"),
        "resnet101": lambda: torchvision.models.resnet101(weights="IMAGENET1K_V1"),
        "resnet152": lambda: torchvision.models.resnet152(weights="IMAGENET1K_V1"),
        "densenet121": lambda: torchvision.models.densenet121(weights="IMAGENET1K_V1"),
        "vgg16": lambda: torchvision.models.vgg16(weights="IMAGENET1K_V1"),
        "convnet": lambda: ConvNet(
            {"channels": cfg.channels, "num_classes": cfg.num_classes}
        ),
    }
    model = model_functions[name]()
    # Modify model for grayscale input if necessary
    if cfg.channels == 1:
        if name.startswith("resnet"):
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif name == "densenet121":
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        elif name == "vgg16":
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )

    # Modify final layer to match the number of classes
    if name.startswith("resnet"):
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, cfg.num_classes)
    elif name == "densenet121":
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, cfg.num_classes)
    elif name == "vgg16":
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, cfg.num_classes)

    return model.cpu()


def _train(tconfig):
    """Train the network on the training set."""
    trainloader = tconfig["train_data"]
    net = tconfig["model"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=tconfig["lr"])
    net = net.to(tconfig["device"]).train()
    epoch_loss = 0
    epoch_acc = 0
    for _epoch in range(tconfig["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(tconfig["device"]), labels.to(tconfig["device"])
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            images = images.cpu()
            labels = labels.cpu()
            # break
        epoch_loss /= total
        epoch_acc = correct / total
    net = net.cpu()
    return {"train_loss": epoch_loss, "train_accuracy": epoch_acc}


def test(net, testloader, device):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    attack_success, attack_total = 0, 0  # New counters for ASR
    net = net.to(device).eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate ASR if the batch contains attacked samples
            if 'target_labels' in batch:
                target_labels = batch['target_labels'].to(device)
                attack_total += labels.size(0)
                attack_success += (predicted == target_labels).sum().item()
            
            images = images.cpu()
            labels = labels.cpu()
    
    loss /= len(testloader.dataset)
    accuracy = correct / total
    asr = attack_success / attack_total if attack_total > 0 else 0.0
    net = net.cpu()
    return {"loss": loss, "accuracy": accuracy, "asr": asr}


def train_neural_network(tconfig):
    """Train the neural network."""
    train_dict = _train(tconfig)
    return train_dict


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
