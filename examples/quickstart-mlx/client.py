import argparse
from collections import OrderedDict

import flwr as fl
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist


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


# Load model and data (simple CNN, CIFAR-10)
num_layers = 2
hidden_dim = 32
num_classes = 10
batch_size = 256
num_epochs = 1
learning_rate = 1e-1

train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=learning_rate)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        layers = model.parameters()["layers"]
        return [np.array(val) for layer in layers for _, val in layer.items()]

    def set_parameters(self, parameters):
        new_params = {}
        new_params["layers"] = [
            {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
            for i in range(0, len(parameters), 2)
        ]
        model.update(new_params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(num_epochs):
            for X, y in batch_iterate(batch_size, train_images, train_labels):
                loss, grads = loss_and_grad_fn(model, X, y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
        return self.get_parameters(config={}), len(train_images), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = eval_fn(model, test_images, test_labels)
        loss = loss_fn(model, test_images, test_labels)
        return loss.item(), len(test_images), {"accuracy": accuracy.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
