"""$project_name: A Flower / $framework_str app."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    partition = fds.load_partition(partition_id)
    partition_splits = partition.train_test_split(test_size=0.2, seed=42)

    partition_splits["train"].set_format("numpy")
    partition_splits["test"].set_format("numpy")

    train_partition = partition_splits["train"].map(
        lambda img: {
            "img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0
        },
        input_columns="image",
    )
    test_partition = partition_splits["test"].map(
        lambda img: {
            "img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0
        },
        input_columns="image",
    )

    data = (
        train_partition["img"],
        train_partition["label"].astype(np.uint32),
        test_partition["img"],
        test_partition["label"].astype(np.uint32),
    )

    train_images, train_labels, test_images, test_labels = map(mx.array, data)
    return train_images, train_labels, test_images, test_labels


def get_params(model):
    layers = model.parameters()["layers"]
    return [np.array(val) for layer in layers for _, val in layer.items()]


def set_params(model, parameters):
    new_params = {}
    new_params["layers"] = [
        {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
        for i in range(0, len(parameters), 2)
    ]
    model.update(new_params)
