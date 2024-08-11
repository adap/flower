"""Flower client example using JAX for linear regression."""

from typing import Callable, Dict, List, Tuple

import flwr as fl
import jax
import jax.numpy as jnp
import jax_training
import numpy as np

# Load data and determine model shape
train_x, train_y, test_x, test_y = jax_training.load_data()
grad_fn = jax.grad(jax_training.loss_fn)
model_shape = train_x.shape[1:]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.params = jax_training.load_model(model_shape)

    def get_parameters(self, config):
        parameters = []
        for _, val in self.params.items():
            parameters.append(np.array(val))
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for key, value in list(zip(self.params.keys(), parameters)):
            self.params[key] = value

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.params, loss, num_examples = jax_training.train(
            self.params, grad_fn, train_x, train_y
        )
        parameters = self.get_parameters(config={})
        return parameters, num_examples, {"loss": float(loss)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, num_examples = jax_training.evaluation(
            self.params, grad_fn, test_x, test_y
        )
        return float(loss), num_examples, {"loss": float(loss)}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080", client=FlowerClient().to_client()
)
