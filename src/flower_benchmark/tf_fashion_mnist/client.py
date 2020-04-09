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
from flower.logger import log

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT

tf.get_logger().setLevel("ERROR")

SEED = 2020
BATCH_SIZE = 50
NUM_EPOCHS = 1
LR_INITIAL = 0.15
LR_DECAY = 0.99


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
    model = load_model(learning_rate=get_lr_initial())
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
        self.epoch = 0
        self.rnd = 0

    def get_parameters(self) -> flwr.ParametersRes:
        parameters = flwr.weights_to_parameters(self.model.get_weights())
        return flwr.ParametersRes(parameters=parameters)

    def fit(self, ins: flwr.FitIns) -> flwr.FitRes:
        weights: flwr.Weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]

        self.rnd += 1
        log(DEBUG, "fit, round %s, config %s", self.rnd, config)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Learning rate
        lr_schedule = get_lr_schedule_rnd(
            self.rnd, lr_initial=LR_INITIAL, lr_decay=LR_DECAY
        )
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # Train the local model using the local dataset
        epochs = num_epochs(self.rnd)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=NUM_EPOCHS,
            callbacks=[lr_scheduler],
            verbose=2,
        )
        self.epoch += epochs

        # Return the refined weights and the number of examples used for training
        return flwr.weights_to_parameters(self.model.get_weights()), len(self.x_train)

    def evaluate(self, ins: flwr.EvaluateIns) -> flwr.EvaluateRes:
        weights = flwr.parameters_to_weights(ins[0])
        config = ins[1]
        log(DEBUG, "evaluate, config %s", config)
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Evaluate the updated model on the local dataset
        loss, _ = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test)
        )
        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.x_test), float(loss)


def num_epochs(rnd: int) -> int:
    """Determine the number of local epochs."""
    if rnd <= 20:
        return 2
    if rnd <= 40:
        return 4
    if rnd <= 60:
        return 6
    return 8


def load_model(
    learning_rate: float, input_shape: Tuple[int, int, int] = (28, 28, 1)
) -> tf.keras.Model:
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
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
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

    # Adjust x sets shape for model
    x_train = adjust_x_shape(x_train)
    x_test = adjust_x_shape(x_test)

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert class vectors to one-hot encoded labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x,y,z) into (x,y,z, 1)."""
    return cast(
        np.ndarray, np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    )


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


def get_lr_schedule_rnd(
    rnd: int, lr_initial: float, lr_decay: float
) -> Callable[[int], float]:
    """Return a schedule which decays the learning rate after each round."""
    lr_rnd = lr_initial * lr_decay ** rnd

    # pylint: disable-msg=unused-argument
    def lr_schedule(epoch: int) -> float:
        """Learning rate schedule."""
        return lr_rnd

    return lr_schedule


def get_lr_initial() -> float:
    """Return the initial learning rate."""
    return get_lr_schedule_rnd(rnd=0, lr_initial=LR_INITIAL, lr_decay=LR_DECAY)(0)


if __name__ == "__main__":
    main()
