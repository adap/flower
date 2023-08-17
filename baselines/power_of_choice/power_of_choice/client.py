"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict
from models import create_MLP_model
from dataset import load_dataset
from flwr.common import Config, Scalar
from omegaconf import DictConfig
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # Return the client's dataset size
        x_entire = np.concatenate((self.x_train, self.x_val))

        properties = {
            "data_size": len(x_entire),
        }

        return properties

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = config["local_epochs"]

        if epochs is None:
            epochs = 2

        batch_size = config["batch_size"]

        if batch_size is None:
            batch_size = 32
            
        fraction_samples = config["fraction_samples"]

        x_train_selected = self.x_train
        y_train_selected = self.y_train

        # Randomly sample num_samples from the training set
        if fraction_samples is not None:
            num_samples = round(len(self.x_train) * fraction_samples)
            idx = np.random.choice(len(self.x_train), num_samples, replace=False)
            x_train_selected = self.x_train[idx]
            y_train_selected = self.y_train[idx]

        print(f"Client training on {len(x_train_selected)} samples, {epochs} epochs, batch size {batch_size}")

        self.model.set_weights(parameters)
        self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        if config["first_phase"]:
            # In the first phase, we evaluate on the entire dataset
            x_entire = np.concatenate((self.x_train, self.x_val))
            y_entire = np.concatenate((self.y_train, self.y_val))
            loss, acc = self.model.evaluate(x_entire, y_entire, verbose=2)
        else:
            # In the normal evaluation phase, we evaluate on the validation set
            loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        return loss, len(self.x_val), {"accuracy": acc}


def gen_client_fn() -> Callable[[str], fl.client.Client]:

    def client_fn(cid: str) -> fl.client.Client:
        # Load model
        model = create_MLP_model()
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid)
    
    return client_fn
