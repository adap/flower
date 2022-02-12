import os

import flwr as fl
import tensorflow as tf
import random
import sys

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self, dropout_prob=0.0):
            self.dropout_prob = dropout_prob

        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            self.check_dropout()
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

        def check_dropout(self):
            # Add this whenever you want to simulate dropout
            r = random.random()
            if r < self.dropout_prob:
                raise Exception("Forced Dropout")

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=CifarClient(dropout_prob=1.0))
