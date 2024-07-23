"""mlxexample: A Flower / MLX app."""

from flwr.common import Context
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from flwr.client import NumPyClient, ClientApp, Client

from mlxexample.task import (
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
    def __init__(self, num_layers, hidden_dim, batch_size, data):
        num_layers = num_layers
        hidden_dim = hidden_dim
        batch_size = batch_size
        num_classes = 10
        num_epochs = 1
        learning_rate = 1e-1

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


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to
    , for example, indicate which dataset to load (e.g accesing the partition-id).
    """

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    num_layers = context.run_config["num_layers"]
    hidden_dim = context.run_config["hidden_dim"]
    batch_size = context.run_config["batch_size"]
    # Return Client instance
    return FlowerClient(num_layers, hidden_dim, batch_size, data).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
