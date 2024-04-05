import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from flwr_datasets import FederatedDataset

import flwr as fl


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


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, model, optim, loss_and_grad_fn, data, num_epochs, batch_size
    ) -> None:
        self.model = model
        self.optimizer = optim
        self.loss_and_grad_fn = loss_and_grad_fn
        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def get_parameters(self, config):
        layers = self.model.parameters()["layers"]
        return [np.array(val) for layer in layers for _, val in layer.items()]

    def set_parameters(self, parameters):
        new_params = {}
        new_params["layers"] = [
            {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
            for i in range(0, len(parameters), 2)
        ]
        self.model.update(new_params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                loss, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return self.get_parameters(config={}), len(self.train_images), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--partition-id",
        choices=[0, 1, 2],
        type=int,
        help="Partition of the dataset divided into 3 iid partitions created artificially.",
    )
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)

    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 1
    learning_rate = 1e-1

    fds = FederatedDataset(dataset="mnist", partitioners={"train": 3})
    partition = fds.load_partition(partition_id=args.partition_id)
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
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            model,
            optimizer,
            loss_and_grad_fn,
            (train_images, train_labels, test_images, test_labels),
            num_epochs,
            batch_size,
        ).to_client(),
    )
