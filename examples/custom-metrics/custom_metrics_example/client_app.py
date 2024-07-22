"""custom_metrics_example: A Flower app for custom metrics."""

import os

import numpy as np
from custom_metrics_example.task import get_model, load_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Method for extra learning metrics calculation
def eval_learning(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(
        y_test, y_pred, average="micro"
    )  # average argument required for multi-class
    prec = precision_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    return acc, rec, prec, f1


# Define Flower client
class FlowerClient(NumPyClient):
    # pylint: disable=too-many-arguments
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        y_pred = self.model.predict(self.x_test)
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
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to
    , for example, indicate which dataset to load (e.g accesing the partition-id).
    """

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    x_train, y_train, x_test, y_test = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    width = int(context.run_config["width"])
    height = int(context.run_config["height"])
    num_channels = int(context.run_config["num_channels"])
    model = get_model(width, height, num_channels)

    # Return Client instance
    return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
