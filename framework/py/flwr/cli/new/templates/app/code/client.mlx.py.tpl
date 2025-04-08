"""$project_name: A Flower / $framework_str app."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import UserConfig
from $import_name.task import (
    MLP,
    batch_iterate,
    eval_fn,
    get_params,
    load_data,
    loss_fn,
    set_params,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self,
        data,
        run_config: UserConfig,
        num_classes,
    ):
        num_layers = run_config["num-layers"]
        hidden_dim = run_config["hidden-dim"]
        input_dim = run_config["input-dim"]
        batch_size = run_config["batch-size"]
        learning_rate = run_config["lr"]
        self.num_epochs = run_config["local-epochs"]

        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.model = MLP(num_layers, input_dim, hidden_dim, num_classes)
        self.optimizer = optim.SGD(learning_rate=learning_rate)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        self.batch_size = batch_size

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                _, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return get_params(self.model), len(self.train_images), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    num_classes = 10

    # Return Client instance
    return FlowerClient(data, context.run_config, num_classes).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
