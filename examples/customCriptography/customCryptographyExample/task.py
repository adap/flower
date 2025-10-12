"""task.py: Modelli e utility per Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.models import resnet18, resnet34


# ----------------------
# MODELLI
# ----------------------
def get_model(model_name: str, num_classes=10, pretrained=False):
    """Restituisce il modello scelto."""
    if model_name == "custom_cnn":
        class CustomCNN(nn.Module):
            """Simple CNN"""
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        return CustomCNN()

    elif model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "resnet34":
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Modello {model_name} non supportato")


# ----------------------
# UTILITY PESI
# ----------------------
def get_weights(net):
    """Estrae i pesi come lista di numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Imposta i pesi a partire da lista di numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ----------------------
# DATA
# ----------------------
def load_data_from_disk(path: str, batch_size: int, resize=None):
    """Carica il dataset dal disco e applica trasformazioni."""
    partition_train_test = load_from_disk(path)

    transforms_list = []
    if resize is not None:
        transforms_list.append(Resize(resize))
    #transforms_list.extend([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transforms_list.extend([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2470, 0.2435, 0.2616])
    ])
    pytorch_transforms = Compose(transforms_list)

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


# ----------------------
# TRAIN & TEST
# ----------------------
def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Addestra il modello."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()

    val_loss, val_acc = test(net, valloader, device)
    return {"val_loss": val_loss, "val_accuracy": val_acc}


def test(net, testloader, device):
    """Valida il modello."""
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
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
