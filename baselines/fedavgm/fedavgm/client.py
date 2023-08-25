"""Define the Flower Client and function to instantiate it."""

import math

import flwr as fl
from keras.utils import to_categorical

from fedavgm.models import create_model


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client."""

    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        input_shape,
        num_classes,
        local_epochs,
        batch_size,
    ) -> None:
        # local model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.new_model()

        # local dataset
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

        # local model params
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def new_model(self):
        """Generate the CNN model input_shape and num_classes."""
        model = create_model(self.input_shape, self.num_classes)
        return model

    def get_parameters(self, config):
        """Return the parameters of the current local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Implement distributed evaluation for a given client."""
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": acc}


def generate_client_fn(partitions, input_shape, num_classes, local_epochs, batch_size):
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        full_x_train_cid, full_y_train_cid = partitions[int(cid)]
        full_y_train_cid = to_categorical(full_y_train_cid, num_classes=num_classes)

        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(full_x_train_cid) * 0.9)
        x_train_cid, y_train_cid = (
            full_x_train_cid[:split_idx],
            full_y_train_cid[:split_idx],
        )
        x_val_cid, y_val_cid = (
            full_x_train_cid[split_idx:],
            full_y_train_cid[split_idx:],
        )

        return FlowerClient(
            x_train_cid,
            y_train_cid,
            x_val_cid,
            y_val_cid,
            input_shape,
            num_classes,
            local_epochs,
            batch_size,
        )

    return client_fn
