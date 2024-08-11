"""mlxexample: A Flower / MLX app."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from mlxexample.task import (
    MLP,
    batch_iterate,
    eval_fn,
    get_params,
    load_data,
    loss_fn,
    set_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, optimizer, batch_size, data):
        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = 1
        self.batch_size = batch_size

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_params(self.model, parameters)
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                _, grads = loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return get_params(self.model), len(self.train_images), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_params(self.model, parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    num_layers = context.run_config["num-layers"]
    hidden_dim = context.run_config["hidden-dim"]
    img_size = context.run_config["img-size"]
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]

    # Prepare model and optimizer
    model = MLP(num_layers, img_size**2, hidden_dim)
    optimizer = optim.SGD(learning_rate=lr)

    # Return Client instance
    return FlowerClient(model, optimizer, batch_size, data).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
