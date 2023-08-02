import os
from sys import argv

import flwr as fl
import tensorflow as tf

SUBSET_SIZE = 1000

if len(argv) > 1:
    transport = argv[1]
else:
    transport = "grpc-bidi"

prefix = "http://" if transport == "rest" else ""

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = x_train[:SUBSET_SIZE], y_train[:SUBSET_SIZE]
x_test, y_test = x_test[:10], y_test[:10]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
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

def client_fn(cid):
    return FlowerClient()

if __name__ == "__main__":
    # Start Flower client
    fl.client.start_numpy_client(server_address=f"{prefix}127.0.0.1:8080", client=FlowerClient(), transport=transport)
