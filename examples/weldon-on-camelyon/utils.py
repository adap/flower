import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def train(net, trainloader, optim):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()
    net.train()
    for X, y in trainloader:
        optim.zero_grad()
        y_pred = net(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optim.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.BCELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        y_pred = []
        y_true = np.array([])
        for X, y in testloader:
            outputs = net(X)
            y_pred.append(outputs)
            y_true = np.append(y_true, y.cpu().numpy())
            loss += criterion(outputs, y).item()

        # Fusion, sigmoid and to numpy
        y_pred = torch.cat(y_pred).numpy()
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0
        acc = accuracy_score(y_true, np.round(y_pred)) if len(set(y_true)) > 1 else 0

    return loss, auc, acc
