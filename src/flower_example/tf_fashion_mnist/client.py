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
from typing import Tuple

import numpy as np
import tensorflow as tf

import flower as fl

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, fashion_mnist


class FashionMnistClient(fl.Client):
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

    def get_parameters(self) -> fl.ParametersRes:
        parameters = fl.weights_to_parameters(self.model.get_weights())
        return fl.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.FitIns) -> fl.FitRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config = ins[1]

        # Get training
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Train the local model using the local dataset
        self.model.fit(
            self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=2
        )

        # Return the refined weights and the number of examples used for training
        weights_prime = fl.weights_to_parameters(self.model.get_weights())
        num_examples = len(self.x_train)
        return weights_prime, num_examples

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:
        weights = fl.parameters_to_weights(ins[0])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, _ = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test), verbose=2
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.x_test), float(loss)


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
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--partition", type=int, required=True, help="Partition index (no default)"
    )
    parser.add_argument(
        "--clients", type=int, required=True, help="Number of clients (no default)",
    )
    args = parser.parse_args()

    # Load model and data
    model = fashion_mnist.load_model()
    xy_train, xy_test = fashion_mnist.load_data(
        partition=args.partition, num_partitions=args.clients
    )

    # Start client
    client = FashionMnistClient(args.cid, model, xy_train, xy_test)
    fl.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


if __name__ == "__main__":
    main()
