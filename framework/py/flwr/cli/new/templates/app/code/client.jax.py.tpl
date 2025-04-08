"""$project_name: A Flower / $framework_str app."""

import jax

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from $import_name.task import (
    evaluation,
    get_params,
    load_data,
    load_model,
    loss_fn,
    set_params,
    train,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, input_dim):
        self.train_x, self.train_y, self.test_x, self.test_y = load_data()
        self.grad_fn = jax.grad(loss_fn)
        self.params = load_model((input_dim,))

    def fit(self, parameters, config):
        set_params(self.params, parameters)
        self.params, loss, num_examples = train(
            self.params, self.grad_fn, self.train_x, self.train_y
        )
        return get_params(self.params), num_examples, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        set_params(self.params, parameters)
        loss, num_examples = evaluation(
            self.params, self.grad_fn, self.test_x, self.test_y
        )
        return float(loss), num_examples, {"loss": float(loss)}


def client_fn(context: Context):
    input_dim = context.run_config["input-dim"]

    # Return Client instance
    return FlowerClient(input_dim).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
