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
from typing import Callable, Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import configure, log
from flower_benchmark.dataset import tf_fashion_mnist_partitioned

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT
from .fashion_mnist import build_dataset, custom_fit, keras_evaluate

tf.get_logger().setLevel("ERROR")

SEED = 2020


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

    # Load model and data
    model = load_model()
    xy_train, xy_test = load_data(
        partition=args.partition, num_clients=args.clients, dry_run=args.dry_run
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


def load_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    """Load model for Fashion-MNIST."""
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(
        512, kernel_initializer=kernel_initializer, activation="relu"
    )(layers)

    outputs = tf.keras.layers.Dense(
        10, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
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
    partition: int, num_clients: int, dry_run: bool
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load partition of randomly shuffled Fashion-MNIST subset."""
    # Load training and test data (ignoring the test data for now)
    xy_partitions, (x_test, y_test) = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=0.0, num_partitions=num_clients
    )

    # Take partition
    x_train, y_train = xy_partitions[partition]

    # Take a subset
    x_test, y_test = shuffle(x_test, y_test, seed=SEED)
    x_test, y_test = get_partition(x_test, y_test, partition, num_clients)

    # Adjust x sets shape for model
    x_train = adjust_x_shape(x_train)
    x_test = adjust_x_shape(x_test)

    # Return a small subset of the data if dry_run is set
    if dry_run:
        return (x_train[0:100], y_train[0:100]), (x_test[0:50], y_test[0:50])
    return (x_train, y_train), (x_test, y_test)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


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
