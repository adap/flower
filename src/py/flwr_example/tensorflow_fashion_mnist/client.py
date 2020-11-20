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
from typing import Dict, Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common import Weights

from . import DEFAULT_SERVER_ADDRESS, fashion_mnist


class FashionMnistClient(fl.client.KerasClient):
    """Flower KerasClient implementing Fashion-MNIST image classification."""

    def __init__(
        self,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ):
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test

    def get_weights(self) -> Weights:
        return cast(Weights, self.model.get_weights())

    def fit(self, weights: Weights, config: Dict[str, str]) -> Tuple[Weights, int, int]:
        # Use provided weights to update local model
        self.model.set_weights(weights)

        # Train the local model using local dataset
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
        )

        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_train), len(self.x_train)

    def evaluate(
        self, weights: Weights, config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Update local model and evaluate on local dataset
        self.model.set_weights(weights)
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test), verbose=2
        )

        # Return number of evaluation examples and evaluation result (loss/accuracy)
        return len(self.x_test), float(loss), float(accuracy)


def main() -> None:
    """Load data, create and start FashionMnistClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--partition", type=int, required=True, help="Partition index (no default)"
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Number of clients (no default)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.partition}", host=args.log_host)

    # Load model and data
    model = fashion_mnist.load_model()
    xy_train, xy_test = fashion_mnist.load_data(
        partition=args.partition, num_partitions=args.clients
    )

    # Start client
    client = FashionMnistClient(model, xy_train, xy_test)
    fl.client.start_keras_client(args.server_address, client)


if __name__ == "__main__":
    main()
