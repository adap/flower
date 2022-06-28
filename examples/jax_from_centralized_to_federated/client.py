"""Flower client example using JAX for linear regression."""

from typing import Dict, List, Tuple, Callable

import flwr as fl
import numpy as np
import jax
import jax.numpy as jnp

import jax_training


class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing using linear regression using JAX."""

    def __init__(
        self,
        params: Dict,
        grad_fn: Callable,
        train_x: List[np.ndarray],
        train_y: List[np.ndarray],
        test_x: List[np.ndarray],
        test_y: List[np.ndarray],
    ) -> None:
        self.params = params
        self.grad_fn = grad_fn
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays
        parameter_value = []
        for _, val in self.params.items():
            parameter_value.append(np.array(val))
        return parameter_value

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Collect model parameters and update the parameters of the local model
        value = jnp.ndarray
        params_item = list(zip(self.params.keys(), parameters))
        for item in params_item:
            key = item[0]
            value = item[1]
            self.params[key] = value
        return self.params

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        print("Start local training")
        self.params = self.set_parameters(parameters)
        self.params, loss, num_examples = jax_training.train(
            self.params, self.grad_fn, self.train_x, self.train_y
        )
        results = {"loss": float(loss)}
        print("Training results", results)
        return self.get_parameters(), num_examples, results

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        print("Start evaluation")
        self.params = self.set_parameters(parameters)
        loss, num_examples = jax_training.evaluation(
            self.params, self.grad_fn, self.test_x, self.test_y
        )
        print("Evaluation accuracy & loss", loss)
        return (
            float(loss),
            num_examples,
            {"loss": float(loss)},
        )


def main() -> None:
    """Load data, start FlowerClient."""

    # Load data
    train_x, train_y, test_x, test_y = jax_training.load_data()
    grad_fn = jax.grad(jax_training.loss_fn)

    # Load model (from centralized training) and initialize parameters
    model_shape = train_x.shape[1:]
    params = jax_training.load_model(model_shape)

    # Start Flower client
    client = FlowerClient(params, grad_fn, train_x, train_y, test_x, test_y)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
