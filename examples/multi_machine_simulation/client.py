import sys
import os

import flwr as fl
import numpy as np
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

server_address = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0:8080"

def load_partition():
    """Load partition from disk."""
    x_train = np.load("./partitions/x_train.npy")
    y_train = np.load("./partitions/y_train.npy")
    x_test = np.load("./partitions/x_test.npy")
    y_test = np.load("./partitions/y_test.npy")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = load_partition()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address, client=CifarClient())
