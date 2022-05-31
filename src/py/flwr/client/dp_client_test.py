import numpy as np
import torch
from opacus import PrivacyEngine
from torch.nn import BCELoss, Linear, Module
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from flwr.client import DPClient


class LogisticRegression(Module):
    """A simple model for testing."""

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(2, 1)

    def forward(self, x):
        """Forward pass."""
        return torch.sigmoid(self.linear(x))


def get_dataloaders(batch_size):
    """Get dataloaders for unit testing."""
    X = torch.from_numpy(
        np.array(
            [
                [1.00000000e00, 0.00000000e00],
                [1.76604444e00, -1.42787610e-01],
                [1.17364818e00, -4.84807753e-01],
                [9.39692621e-01, 3.42020143e-01],
                [-9.39692621e-01, 3.42020143e-01],
                [-1.73648178e-01, 9.84807753e-01],
                [6.03073792e-02, 1.57979857e-01],
                [5.00000000e-01, 8.66025404e-01],
                [1.93969262e00, 1.57979857e-01],
                [1.50000000e00, -3.66025404e-01],
                [5.00000000e-01, -3.66025404e-01],
                [7.66044443e-01, 6.42787610e-01],
                [-1.00000000e00, 1.22464680e-16],
                [2.00000000e00, 5.00000000e-01],
                [1.73648178e-01, 9.84807753e-01],
                [2.33955557e-01, -1.42787610e-01],
                [-7.66044443e-01, 6.42787610e-01],
                [0.00000000e00, 5.00000000e-01],
                [8.26351822e-01, -4.84807753e-01],
                [-5.00000000e-01, 8.66025404e-01],
            ]
        ).astype("float32")
    )
    y = torch.from_numpy(
        np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]).astype("float32")
    ).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return train_loader, test_loader


def test_dp_client_init():
    """DPClient can be constructed correctly."""
    module = LogisticRegression()
    privacy_engine = PrivacyEngine()
    batch_size = 4
    train_loader, test_loader = get_dataloaders(batch_size)
    optimizer = SGD(module.parameters(), lr=0.001)
    criterion = BCELoss()
    target_epsilon = 1.0
    target_delta = 0.1
    client = DPClient(
        module,
        optimizer,
        criterion,
        privacy_engine,
        train_loader,
        test_loader,
        target_epsilon,
        target_delta,
        epochs=10,
        max_grad_norm=1.0,
    )
    assert client.privacy_engine.get_epsilon(target_delta) == 0.0


def test_dp_client_fit():
    module = LogisticRegression()
    privacy_engine = PrivacyEngine()
    batch_size = 4
    train_loader, test_loader = get_dataloaders(batch_size)
    optimizer = SGD(module.parameters(), lr=0.001)
    criterion = BCELoss()
    target_epsilon = 1.0
    target_delta = 0.1

    def accuracy(predictions, actuals):
        total = actuals.size(0)
        correct = (predictions.eq(actuals)).sum().item()
        return correct / total

    client = DPClient(
        module,
        optimizer,
        criterion,
        privacy_engine,
        train_loader,
        test_loader,
        target_epsilon,
        target_delta,
        epochs=10,
        max_grad_norm=1.0,
        accuracy=accuracy,
    )

    parameters, num_examples, metrics = client.fit(client.get_parameters(), {})
    assert parameters is not None
    assert num_examples == 20
    assert metrics["epsilon"] is not None
    assert metrics["accept"] is not None
    assert metrics["accuracy"] > 0.0 and metrics["accuracy"] < 1.0


def test_dp_client_evaluate():
    """DPClient.evaluate() works with an accuracy score function."""
    module = LogisticRegression()
    privacy_engine = PrivacyEngine()
    batch_size = 4
    train_loader, test_loader = get_dataloaders(batch_size)
    optimizer = SGD(module.parameters(), lr=0.001)
    criterion = BCELoss()
    target_epsilon = 1.0
    target_delta = 0.1

    def accuracy(predictions, actuals):
        """Binary classifier accuracy score function."""
        total = actuals.size(0)
        correct = (predictions.gt(0.5) == actuals).sum().item()
        return correct / total

    client = DPClient(
        module,
        optimizer,
        criterion,
        privacy_engine,
        train_loader,
        test_loader,
        target_epsilon,
        target_delta,
        epochs=10,
        max_grad_norm=1.0,
        accuracy=accuracy,
    )
    loss, num_examples, metrics = client.evaluate(client.get_parameters(), {})
    assert loss > 0
    assert num_examples == 20
    assert metrics["accuracy"] > 0.0 and metrics["accuracy"] < 1.0
