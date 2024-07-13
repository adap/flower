"""$project_name: A Flower / JAX app."""

import jax
from flwr.client import NumPyClient, ClientApp
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
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = load_data()
        self.grad_fn = jax.grad(loss_fn)
        model_shape = self.train_x.shape[1:]

        self.params = load_model(model_shape)

    def get_parameters(self, config):
        return get_params(self.params)

    def set_parameters(self, parameters):
        set_params(self.params, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.params, loss, num_examples = train(
            self.params, self.grad_fn, self.train_x, self.train_y
        )
        parameters = self.get_parameters(config={})
        return parameters, num_examples, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, num_examples = evaluation(
            self.params, self.grad_fn, self.test_x, self.test_y
        )
        return float(loss), num_examples, {"loss": float(loss)}

def client_fn(context: Context):
    # Return Client instance
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
