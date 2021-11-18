"""Flower client example using MXNet for MNIST classification."""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import jax
import jax.numpy as jnp

import jax_training


# Flower Client
class MNISTClient(fl.client.NumPyClient):
    """Flower client implementing MNIST classification using MXNet."""

    def __init__(
        self,
        params,
        grad_fn,
        train_x ,
        train_y ,
        test_x,
        test_y,
    ) -> None:
        self.params= params
        self.grad_fn = grad_fn
        self.train_x = train_x
        self. train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays,
        parameter_value = []
        for _, val in self.params.items():
            parameter_value.append(np.array(val))
        return parameter_value
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Collect model parameters and set new weight values
        value=jnp.ndarray
        params_item = list(zip(self.params.keys(),parameters))
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
        #print("Set Parameter before Training", self.params)
        self.params, loss, num_examples = jax_training.train(self.params, self.grad_fn, self.train_x, self.train_y)
        #print("Set Parameter after Training", self.params)
        results = {"loss": float(loss)}
        print("Training results", results)
        #parameter_value = []
        #for _, val in self.params.items():
        #    parameter_value.append(np.array(val))
        return self.get_parameters(), num_examples, results

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[int, float, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        print("Start evaluation")
        self.params = self.set_parameters(parameters)
        #print("Parameter for Evaluation", self.params)
        loss, num_examples = jax_training.evaluation(self.params,self.grad_fn, self.test_x, self.test_y)
        print("Evaluation accuracy & loss", loss)
        return (
            float(loss),
            num_examples,
            {"loss": float(loss)},
        )


def main() -> None:
    """Load data, start MNISTClient."""

    # Load data
    train_x, train_y, test_x, test_y = jax_training.load_data()
    grad_fn = jax.grad(jax_training.loss_fn)

    # Load model (from centralized training) and initialize parameters
    model_shape = train_x.shape[1:]
    params = jax_training.load_model(model_shape)

    # Start Flower client
    client = MNISTClient(params, grad_fn, train_x, train_y, test_x, test_y)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
