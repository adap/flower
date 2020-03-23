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
from logging import DEBUG
from typing import Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import log

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT

tf.get_logger().setLevel("ERROR")

SEED = 2020
BATCH_SIZE = 50
NUM_EPOCHS = 5


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
    parser.add_argument("--partition", type=int, help="Partition index (no default)")
    parser.add_argument(
        "--clients", type=int, help="Number of clients (no default)",
    )
    args = parser.parse_args()

    # Load model and data
    model = load_model()

    xy_train, xy_test = load_data(partition=args.partition, num_clients=args.clients)

    # Start client
    client = FashionMnistClient(args.cid, model, xy_train, xy_test)
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


class FashionMnistClient(flwr.Client):
    """Flower client implementing Fashion-MNIST image classification using TensorFlow/Keras."""

    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ):
        super().__init__(cid)
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test

    def get_weights(self) -> flwr.Weights:
        return cast(flwr.Weights, self.model.get_weights())

    def fit(self, weights: flwr.Weights) -> Tuple[flwr.Weights, int]:
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Train the local model using the local dataset
        self.model.fit(self.x_train, self.y_train, epochs=NUM_EPOCHS, verbose=2)
        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_train)

    def evaluate(self, weights: flwr.Weights) -> Tuple[int, float]:
        log(DEBUG, "evaluate")
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Evaluate the updated model on the local dataset
        loss, _ = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test)
        )
        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.x_test), float(loss)


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


def load_data(
    partition: int, num_clients: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load partition of randomly shuffled Fashion-MNIST subset."""
    # Load training and test data (ignoring the test data for now)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Take a subset
    x_train, y_train = shuffle(x_train, y_train, seed=SEED)
    x_test, y_test = shuffle(x_test, y_test, seed=SEED)

    x_train, y_train = get_partition(x_train, y_train, partition, num_clients)
    x_test, y_test = get_partition(x_test, y_test, partition, num_clients)

    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def shuffle(
    x_orig: np.ndarray, y_orig: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    return x_orig[idx], y_orig[idx]


def get_partition(
    x_orig: np.ndarray, y_orig: np.ndarray, partition: int, num_clients: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single partition of an equally partitioned dataset."""
    step_size = len(x_orig) / num_clients
    start_index = int(step_size * partition)
    end_index = int(start_index + step_size)
    return x_orig[start_index:end_index], y_orig[start_index:end_index]


if __name__ == "__main__":
    main()
