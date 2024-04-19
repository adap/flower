"""$project_name: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr_datasets import FederatedDataset


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


fds = FederatedDataset(dataset="cifar10", partitioners={"train": 2})

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""

    # Load model and data (MobileNetV2, CIFAR-10)
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Download and partition dataset
    partition = fds.load_partition(int(cid), "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    train_data = partition["train"]["img"] / 255.0, partition["train"]["label"]
    test_data = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return FlowerClient(model, train_data, test_data).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
