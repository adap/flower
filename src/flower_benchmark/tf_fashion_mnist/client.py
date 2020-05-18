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
"""Flower client using TensorFlow for Fashion-MNIST image classification."""


import argparse
from logging import DEBUG
from typing import Tuple

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import configure, log
from flower_benchmark.common import (
    build_dataset,
    custom_fit,
    keras_evaluate,
    load_partition,
)
from flower_benchmark.dataset import tf_fashion_mnist_partitioned
from flower_benchmark.model import orig_cnn

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, SEED

tf.get_logger().setLevel("ERROR")


def main() -> None:
    """Load data, create and start FashionMnistClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (IPv6, default: [::])",
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
    parser.add_argument(
        "--delay_factor",
        type=float,
        default=0.0,
        help="Delay factor increases the time batches take to compute (default: 0.0)",
    )
    parser.add_argument(
        "--dry_run", type=bool, default=False, help="Dry run (default: False)"
    )
    parser.add_argument(
        "--log_file", type=str, help="Log file path (no default)",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    configure(f"client:{args.cid}", args.log_file, args.log_host)

    # Load model
    model = orig_cnn(input_shape=(28, 28, 1), seed=SEED)

    # Load local data partition
    xy_partitions, xy_test = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=0.0, num_partitions=args.clients
    )
    xy_train, xy_test = load_partition(
        xy_partitions,
        xy_test,
        partition=args.partition,
        num_clients=args.clients,
        dry_run=args.dry_run,
        seed=SEED,
    )

    # Start client
    client = FashionMnistClient(
        args.cid, model, xy_train, xy_test, args.delay_factor, 10
    )
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


class FashionMnistClient(flwr.Client):
    """Flower client implementing Fashion-MNIST image classification using TensorFlow/Keras."""

    # pylint: disable-msg=too-many-arguments
    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        delay_factor: float,
        num_classes: int,
    ):
        super().__init__(cid)
        self.model = model
        self.ds_train = build_dataset(
            xy_train[0],
            xy_train[1],
            num_classes=num_classes,
            shuffle_buffer_size=len(xy_train[0]),
            augment=False,
        )
        self.ds_test = build_dataset(
            xy_test[0],
            xy_test[1],
            num_classes=num_classes,
            shuffle_buffer_size=0,
            augment=False,
        )
        self.num_examples_train = len(xy_train[0])
        self.num_examples_test = len(xy_test[0])
        self.delay_factor = delay_factor

    def get_parameters(self) -> flwr.ParametersRes:
        parameters = flwr.weights_to_parameters(self.model.get_weights())
        return flwr.ParametersRes(parameters=parameters)

    def fit(self, ins: flwr.FitIns) -> flwr.FitRes:
        weights: flwr.Weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]
        log(
            DEBUG,
            "fit on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_train,
            config,
        )

        # Training configuration
        # epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        # lr_initial = float(config["lr_initial"])
        # lr_decay = float(config["lr_decay"])
        timeout = int(config["timeout"])
        partial_updates = bool(int(config["partial_updates"]))

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Train the local model using the local dataset
        completed, fit_duration, num_examples = custom_fit(
            model=self.model,
            dataset=self.ds_train,
            num_epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            delay_factor=self.delay_factor,
            timeout=timeout,
        )
        log(DEBUG, "client %s had fit_duration %s", self.cid, fit_duration)

        # Compute the maximum number of examples which could have been processed
        num_examples_ceil = self.num_examples_train * epochs

        # Return empty update if local update could not be completed in time
        if not completed and not partial_updates:
            parameters = flwr.weights_to_parameters([])
            return parameters, num_examples, num_examples_ceil

        # Return the refined weights and the number of examples used for training
        parameters = flwr.weights_to_parameters(self.model.get_weights())
        return parameters, num_examples, num_examples_ceil

    def evaluate(self, ins: flwr.EvaluateIns) -> flwr.EvaluateRes:
        weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]
        log(
            DEBUG,
            "evaluate on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_test,
            config,
        )

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, _ = keras_evaluate(
            self.model, self.ds_test, batch_size=self.num_examples_test
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return self.num_examples_test, loss


if __name__ == "__main__":
    main()
