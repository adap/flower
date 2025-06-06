import os

import numpy as np
import tensorflow as tf
from datasets import load_dataset

from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context

SUBSET_SIZE = 1000

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load CIFAR-10 from Hugging Face
dataset = load_dataset("uoft-cs/cifar10")

# Convert to NumPy arrays
x_train = np.stack(dataset["train"]["img"]).astype("float32") / 255.0
y_train = np.array(dataset["train"]["label"])

x_test = np.stack(dataset["test"]["img"]).astype("float32") / 255.0
y_test = np.array(dataset["test"]["label"])

x_train, y_train = x_train[:SUBSET_SIZE], y_train[:SUBSET_SIZE]
x_test, y_test = x_test[:10], y_test[:10]

# Load model (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), classes=10, weights=None
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
