import torch 
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import Compose
from typing import Union
from typing import List
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter

def train_local(net: nn.Module, 
                trainloader: DataLoader, 
                epochs: int,
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                DEVICE: torch.device):
    """Train the network on the training set."""
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)  # dict, non tupla
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if batch_idx % 100 == 0:
                print(f"Image number {total} of {len(trainloader.dataset)}", end="\r", flush=True)

        # Fuori dal loop sui batch
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}", end="\r", flush=True)

