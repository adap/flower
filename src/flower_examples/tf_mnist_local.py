# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Minimal example on how to use the Flower framework with
  - TensorFlow 2.0+ (Keras)
  - MNIST image classification
"""
from typing import Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import log


def load_model() -> tf.keras.Model:
    """Create simple fully-connected neural network"""
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
    return model


class MnistClient(flwr.Client):
    """Flower client implementing MNIST image classification using TensorFlow/Keras."""

    def __init__(
        self, cid: str, model: tf.keras.Model, x_local: np.ndarray, y_local: np.ndarray
    ):
        super().__init__(cid)
        self.model = model
        self.x_local = x_local
        self.y_local = y_local

    def get_weights(self) -> flwr.Weights:
        return cast(flwr.Weights, self.model.get_weights())

    def fit(self, weights: flwr.Weights) -> Tuple[flwr.Weights, int]:
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Train the local model using the local dataset
        self.model.fit(self.x_local, self.y_local, epochs=4, verbose=2)
        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_local)

    def evaluate(self, weights: flwr.Weights) -> Tuple[int, float]:
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Evaluate the updated model on the local dataset
        loss, _ = self.model.evaluate(self.x_local, self.y_local, verbose=0)
        # Return the number of examples used for evaluation along with the evaltion result (loss)
        return len(self.x_local), float(loss)


def main() -> None:  # pylint: disable-msg=too-many-locals
    """Basic Flower usage:
        1. Create clients (each holding their respective local data partition)
        2. Create server
        3. Start learning
    """

    # Load training and test data
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0

    # Split the data into three partitions (holding 20k examples and labels each) to
    # simulate three distinct datasets
    x_0, y_0 = x_train[:20000], y_train[:20000]
    x_1, y_1 = x_train[20000:40000], y_train[20000:40000]
    x_2, y_2 = x_train[40000:], y_train[40000:]

    # Create three clients (each holding their own dataset)
    c_0 = MnistClient("c0", load_model(), x_0, y_0)
    c_1 = MnistClient("c1", load_model(), x_1, y_1)
    c_2 = MnistClient("c2", load_model(), x_2, y_2)

    # Create ClientManager and register clients
    mngr = flwr.SimpleClientManager()
    mngr.register(c_0)
    mngr.register(c_1)
    mngr.register(c_2)

    # Start server and train four rounds
    server = flwr.Server(client_manager=mngr)
    hist = server.fit(num_rounds=2)
    log("DEBUG", f"Flower training: {hist}")

    # Evaluate the final trained model
    loss = server.evaluate()
    log("DEBUG", f"Final loss after training: {loss}")


if __name__ == "__main__":
    main()
