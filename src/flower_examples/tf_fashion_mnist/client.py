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
"""Flower client example using TensorFlow for Fashion-MNIST image classification."""


import argparse
from typing import Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT

tf.get_logger().setLevel("ERROR")


def main() -> None:
    """Load data, create and start FashionMnistClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (default: [::])",
    )
    parser.add_argument(
        "--grpc_server_port",
        type=int,
        default=DEFAULT_GRPC_SERVER_PORT,
        help="gRPC server port (default: 8080)",
    )
    parser.add_argument("--cid", type=str, help="Client CID (no default)")
    args = parser.parse_args()

    # Load data and start client
    x_local, y_local = load_data()
    client = FashionMnistClient(args.cid, load_model(), x_local, y_local)
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


class FashionMnistClient(flwr.Client):
    """Flower client implementing Fashion-MNIST image classification using TensorFlow/Keras."""

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


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load random Fashion-MNIST subset."""
    # Load training and test data (ignoring the test data for now)
    (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0

    # Take a random subset of the dataset to simulate different local datasets
    idx = np.random.choice(np.arange(len(x_train)), 20000, replace=False)
    x_sample, y_sample = x_train[idx], y_train[idx]

    # Return the random subset
    return x_sample, y_sample


if __name__ == "__main__":
    main()
