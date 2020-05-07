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
"""Flower client using TensorFlow for CIFAR-10/100."""


import argparse
from logging import DEBUG
from typing import Callable, Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import log
from flower_benchmark.dataset import tf_cifar_partitioned

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT
from .cifar import build_dataset, keras_evaluate, keras_fit

tf.get_logger().setLevel("ERROR")

SEED = 2020


def main() -> None:
    """Load data, create and start CifarClient."""
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
        "--cifar",
        type=int,
        choices=[10, 100],
        default=10,
        help="CIFAR version, allowed values: 10 or 100 (default: 10)",
    )
    parser.add_argument(
        "--clients", type=int, required=True, help="Number of clients (no default)",
    )
    parser.add_argument(
        "--dry_run", type=bool, default=False, help="Dry run (default: False)"
    )
    args = parser.parse_args()

    # Load model and data
    model = load_model(input_shape=(32, 32, 3), num_classes=args.cifar)
    xy_train, xy_test = load_data(
        partition=args.partition,
        num_classes=args.cifar,
        num_clients=args.clients,
        dry_run=args.dry_run,
    )

    # Start client
    client = CifarClient(args.cid, args.cifar, model, xy_train, xy_test)
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


class CifarClient(flwr.Client):
    """Flower client implementing CIAFR-10/100 image classification using TF."""

    # pylint: disable-msg=too-many-arguments
    def __init__(
        self,
        cid: str,
        num_classes: int,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ) -> None:
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
            shuffle_buffer_size=len(xy_test[0]),
            augment=False,
        )
        self.num_examples_train = len(xy_train[0])
        self.num_examples_test = len(xy_test[0])

    def get_parameters(self) -> flwr.ParametersRes:
        parameters = flwr.weights_to_parameters(self.model.get_weights())
        return flwr.ParametersRes(parameters=parameters)

    def fit(self, ins: flwr.FitIns) -> flwr.FitRes:
        weights: flwr.Weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]
        log(DEBUG, "fit, config %s", config)

        # Training configuration
        # epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        # lr_initial = float(config["lr_initial"])
        # lr_decay = float(config["lr_decay"])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Train the local model using the local dataset
        keras_fit(
            model=self.model,
            dataset=self.ds_train,
            num_epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
        )

        # Compute the maximum number of examples which could have been processed
        num_examples_ceil = self.num_examples_train * epochs

        # Return the refined weights and the number of examples used for training
        parameters = flwr.weights_to_parameters(self.model.get_weights())
        num_examples = self.num_examples_train
        return parameters, num_examples, num_examples_ceil

    def evaluate(self, ins: flwr.EvaluateIns) -> flwr.EvaluateRes:
        weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]
        log(DEBUG, "evaluate, config %s", config)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, _ = keras_evaluate(
            self.model, self.ds_test, batch_size=self.num_examples_test
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return self.num_examples_test, loss


def load_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Create a ResNet-50 (v2) instance"""
    model = tf.keras.applications.ResNet50V2(
        weights=None, include_top=True, input_shape=input_shape, classes=num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_lr_schedule(
    epoch_global: int, lr_initial: float, lr_decay: float
) -> Callable[[int], float]:
    """Return a schedule which decays the learning rate after each epoch."""

    def lr_schedule(epoch: int) -> float:
        """Learning rate schedule."""
        epoch += epoch_global
        return lr_initial * lr_decay ** epoch

    return lr_schedule


def load_data(
    partition: int, num_classes: int, num_clients: int, dry_run: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load, normalize, and sample CIFAR-10/100."""
    use_cifar100 = num_classes == 100
    xy_partitions, (x_test, y_test) = tf_cifar_partitioned.load_data(
        iid_fraction=0.9, num_partitions=num_clients, cifar100=use_cifar100
    )
    x_train, y_train = xy_partitions[partition]

    log(DEBUG, "Data distribution %s", np.unique(y_train, return_counts=True))

    y_train = adjust_y_shape(y_train)
    y_test = adjust_y_shape(y_test)

    # Return a small subset of the data if dry_run is set
    if dry_run:
        return (x_train[0:100], y_train[0:100]), (x_test[0:50], y_test[0:50])
    return (x_train, y_train), (x_test, y_test)


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return cast(np.ndarray, nda_adjusted)


if __name__ == "__main__":
    main()
