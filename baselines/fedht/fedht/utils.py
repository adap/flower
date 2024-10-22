"""Execute utility functions for fedht baseline."""

import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Create an object with Dataset parent class for dataloader."""

    def __init__(self, features, labels):
        """Initialize self with data object."""
        self.data = {"image": features, "label": labels}

    def __len__(self):
        """Return length of features in data."""
        return len(self.data["image"])

    def __getitem__(self, idx):
        """Assign key."""
        return {key: value[idx] for key, value in self.data.items()}


def partition_data(data, num_partitions):
    """Partition data into clients."""
    # Calculate the size of each partition
    X, y = data
    partition_size = len(X) // num_partitions
    # Create partitions
    partitions_x = [
        X[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)
    ]
    partitions_y = [
        y[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)
    ]

    # Handle any remaining items
    if len(data) % num_partitions != 0:
        # partitions[-1] = partitions[-1] + data[num_partitions * partition_size:]
        partitions_x[-1] = np.vstack(
            (partitions_x[-1], X[num_partitions * partition_size :])
        )
        partitions_y[-1] = np.vstack(
            (partitions_y[-1], y[num_partitions * partition_size :])
        )

    return partitions_x, partitions_y


def sim_data(ni: int, num_clients: int, num_features: int, alpha=1, beta=1):
    """Simulate data for simII."""
    # generate client-based model coefs
    u = np.random.normal(0, alpha, num_clients)
    x = np.zeros((num_features, num_clients))
    x[0:99, :] = np.random.multivariate_normal(u, np.diag(np.ones(num_clients)), 99)

    # generate observations
    ivec = np.arange(1, num_features + 1)
    vari = np.diag(1 / (ivec**1.2))

    B = np.random.normal(0, beta, num_features)
    v = np.random.multivariate_normal(B, np.diag(np.ones(num_features)), num_clients)

    error = np.random.multivariate_normal(u, np.diag(np.ones(num_clients)), ni)
    z = np.zeros((num_clients, ni, num_features))
    y = np.zeros((ni, num_clients))

    # (num_clients, ni, num_feaures)
    for i in range(z.shape[0]):
        z[i, :, :] = np.random.multivariate_normal(v[i], vari, ni)
        hold = np.matmul(z[i, :, :], x[:, i]) + error[:, i]
        y[:, i] = np.exp(hold) / (1 + np.exp(hold))

    for j in range(num_clients):
        top_indices = np.argpartition(y[:, j], -100)[-100:]
        mask = np.zeros(y[:, j].shape, dtype=bool)
        mask[top_indices] = True
        y[mask, j] = 1
        y[~mask, j] = 0

    # might need to adjust; vague data generating process
    return z, y
