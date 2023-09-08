from typing import Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl

### uncomment this if you are getting the ssl error
# ssl._create_default_https_context = ssl._create_unverified_context
###


def main() -> None:
    # Build and compile Keras model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Implement a Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config) -> fl.common.NDArrays:
            return cast(fl.common.NDArrays, model.get_weights())

        def fit(self, parameters, config) -> Tuple[fl.common.NDArrays, int, dict]:
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config) -> Tuple[int, int, dict]:
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MnistClient(),
    )


if __name__ == "__main__":
    main()
