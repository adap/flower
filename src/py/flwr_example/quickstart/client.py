from typing import Dict, Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl


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

    # Implement a Flower client
    class MnistClient(fl.client.keras_client.KerasClient):
        def __init__(
            self,
            cid: str,
            model: tf.keras.Model,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
        ) -> None:
            super().__init__(cid)
            self.model = model
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test

        def get_weights(self) -> fl.common.Weights:
            return cast(fl.common.Weights, self.model.get_weights())

        def fit(
            self, weights: fl.common.Weights, config: Dict[str, str]
        ) -> Tuple[fl.common.Weights, int, int]:
            self.model.set_weights(weights)
            self.model.fit(self.x_train, self.y_train, epochs=5)
            return self.model.get_weights(), len(self.x_train), len(self.x_train)

        def evaluate(
            self, weights: fl.common.Weights, config: Dict[str, str]
        ) -> Tuple[int, float, float]:
            self.model.set_weights(weights)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
            return len(self.x_test), loss, accuracy

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Instanstiate client
    client = MnistClient("0", model, x_train, y_train, x_test, y_test)

    # Start client
    fl.client.start_keras_client(server_address="[::]:8080", client=client)


if __name__ == "__main__":
    main()
