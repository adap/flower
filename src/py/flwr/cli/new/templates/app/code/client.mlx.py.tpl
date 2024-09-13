"""$project_name: A Flower / $framework_str app."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from $import_name.task import (
    batch_iterate,
    eval_fn,
    get_params,
    load_data,
    loss_fn,
    set_params,
    MLP,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self,
        data,
        num_layers,
        hidden_dim,
        num_classes,
        batch_size,
        learning_rate,
        num_epochs,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.model = MLP(
            num_layers, self.train_images.shape[-1], hidden_dim, num_classes
        )
        self.optimizer = optim.SGD(learning_rate=learning_rate)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def get_parameters(self, config):
        return get_params(self.model)

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                _, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return self.get_parameters(config={}), len(self.train_images), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    num_layers = context.run_config["num-layers"]
    hidden_dim = context.run_config["hidden-dim"]
    num_classes = 10
    batch_size = context.run_config["batch-size"]
    learning_rate = context.run_config["lr"]
    num_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(
        data, num_layers, hidden_dim, num_classes, batch_size, learning_rate, num_epochs
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
