import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader as dl


def get_data(cid, train=True):
    return dl(
        FedHeartDisease(center=cid, train=train),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)


def train(net, trainloader, epochs, use_gpu):
    """Train the model on the training set."""
    loss = BaselineLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for X, y in trainloader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            y_pred = net(X)
            lm = loss(y_pred, y)
            lm.backward()
            optimizer.step()


def test(net, testloader, use_gpu):
    """Validate the model on the test set."""
    if use_gpu:
        net = net.cuda()
    net.eval()
    criterion = BaselineLoss()
    loss = 0.0
    with torch.no_grad():
        y_pred_final = []
        y_true_final = []
        for X, y in testloader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()
            y_pred = net(X).detach().cpu()
            y = y.detach().cpu()
            loss += criterion(y_pred, y).item()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
    net.train()
    return loss / len(testloader.dataset), metric(y_true_final, y_pred_final)


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of rounds to run FL for.",
    )
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() and not (args.cpu_only)

    total_accuracy = 0.0

    for i in range(4):
        net = Baseline()
        train_loader = get_data(i, train=True)
        test_loader = get_data(i, train=False)

        train(net, train_loader, epochs=args.n_epochs, use_gpu=use_gpu)
        loss, accuracy = test(net, test_loader, use_gpu)
        total_accuracy += accuracy

        print("Loss:", loss)
        print("Accuracy:", accuracy)

    print("Average:", total_accuracy / 4)
