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
"""Example on how to build a Flower client using TensorFlow for CIFAR-10/100.

Several components are translated from the official Keras ResNet/CIFAR example:
https://keras.io/examples/cifar10_resnet/
"""

import argparse
from logging import DEBUG
from typing import Callable, Optional, Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr
from flower.logger import log

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT

tf.get_logger().setLevel("ERROR")

SEED = 2020
BATCH_SIZE = 64


def main() -> None:
    """Load data and start CifarClient."""

    # Parse command line arguments
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
        "--cifar",
        type=int,
        choices=[10, 100],
        default=10,
        help="CIFAR version, allowed values: 10 or 100 (default: 10)",
    )
    parser.add_argument(
        "--clients", type=int, help="Number of clients (no default)",
    )
    args = parser.parse_args()
    log(
        DEBUG,
        "Run client, cid %s, partition %s, CIFAR-%s",
        args.cid,
        args.partition,
        args.cifar,
    )

    # Load model and data
    model = load_model(
        input_shape=(32, 32, 3), num_classes=args.cifar, learning_rate=get_lr_initial()
    )
    xy_train, xy_test = load_data(
        partition=args.partition, num_classes=args.cifar, num_clients=args.clients
    )

    # Start client
    client = CifarClient(args.cid, model, xy_train, xy_test)
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


# pylint: disable-msg=too-many-instance-attributes
class CifarClient(flwr.Client):
    """Flower client implementing CIAFR-10/100 image classification using TF."""

    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        super().__init__(cid)
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test
        self.datagen: Optional[tf.keras.preprocessing.image.ImageDataGenerator] = None
        self.epoch = 0
        self.rnd = 0

    def get_weights(self) -> flwr.Weights:
        log(DEBUG, "get_weights")
        return cast(flwr.Weights, self.model.get_weights())

    def fit(self, weights: flwr.Weights) -> Tuple[flwr.Weights, int]:
        self.rnd += 1
        log(DEBUG, "fit, round %s", self.rnd)

        # Lazy initialization of the ImageDataGenerator
        if self.datagen is None:
            self.datagen = load_datagen(self.x_train)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Learning rate
        lr_schedule = get_lr_schedule(self.epoch)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
        )
        callbacks = [lr_reducer, lr_scheduler]

        # Train the local model using the local dataset
        epochs = num_epochs(self.rnd)
        self.model.fit_generator(
            self.datagen.flow(self.x_train, self.y_train, batch_size=BATCH_SIZE),
            epochs=num_epochs(self.rnd),
            callbacks=callbacks,
        )
        self.epoch += epochs

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
        # Return the number of evaluation examples and the evaltion result (loss)
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
    input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float
) -> tf.keras.Model:
    """Create a ResNet-50 (v2) instance"""
    model = tf.keras.applications.ResNet50V2(
        weights=None, include_top=True, input_shape=input_shape, classes=num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(
    partition: int,
    num_classes: int,
    num_clients: int,
    subtract_pixel_mean: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load, normalize, and sample CIFAR-10/100."""
    cifar = (
        tf.keras.datasets.cifar10 if num_classes == 10 else tf.keras.datasets.cifar100
    )
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    # Take a subset
    x_train, y_train = shuffle(x_train, y_train, seed=SEED)
    x_test, y_test = shuffle(x_test, y_test, seed=SEED)

    x_train, y_train = get_partition(x_train, y_train, partition, num_clients)
    x_test, y_test = get_partition(x_test, y_test, partition, num_clients)

    # Normalize data.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Convert class vectors to one-hot encoded labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

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


def load_datagen(
    x_train: np.ndarray,
) -> tf.keras.preprocessing.image.ImageDataGenerator:
    """Create an ImageDataGenerator for CIFAR-10/100."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
    )
    datagen.fit(x_train)
    return datagen


def get_lr_schedule(epoch_base: int) -> Callable[[int], float]:
    """Return a learning rate function."""

    def lr_schedule(epoch: int) -> float:
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        epoch += epoch_base
        learning_rate = 1e-3
        if epoch > 180:
            learning_rate *= 0.5e-3
        elif epoch > 160:
            learning_rate *= 1e-3
        elif epoch > 120:
            learning_rate *= 1e-2
        elif epoch > 80:
            learning_rate *= 1e-1
        return learning_rate

    return lr_schedule


def get_lr_initial() -> float:
    """Return the initial learning rate."""
    return get_lr_schedule(0)(0)


if __name__ == "__main__":
    main()
