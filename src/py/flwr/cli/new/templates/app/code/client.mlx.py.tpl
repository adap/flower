"""$project_name: A Flower / MLX app."""

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
    def __init__(self, data):
        num_layers = 2
        hidden_dim = 32
        num_classes = 10
        batch_size = 256
        num_epochs = 1
        learning_rate = 1e-1

        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.model = MLP(num_layers, self.train_images.shape[-1], hidden_dim, num_classes) 
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
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    data = load_data(partition_id, num_partitions)

    # Return Client instance
    return FlowerClient(data).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
