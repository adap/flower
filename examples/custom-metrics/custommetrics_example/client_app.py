"""custommetrics_example: A Flower / TensorFlow app for custom metrics."""

import os

import numpy as np
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from custommetrics_example.task import eval_learning, get_model, load_data

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class FlowerClient(NumPyClient):
    # pylint: disable=too-many-arguments
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=32, verbose=False
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)
        y_pred = self.model.predict(self.x_test, verbose=False)
        y_pred = np.argmax(y_pred, axis=1).reshape(
            -1, 1
        )  # MobileNetV2 outputs 10 possible classes, argmax returns just the most probable

        acc, rec, prec, f1 = eval_learning(self.y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,  # accuracy from tensorflow model.evaluate
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
        }
        return loss, len(self.x_test), output_dict


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load the train and test data
    x_train, y_train, x_test, y_test = load_data(partition_id, num_partitions)

    model = get_model()

    # Return Client instance
    return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
