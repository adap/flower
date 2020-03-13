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
"""Example on how to build a Flower client using TensorFlow for CIFAR-10/100."""

import argparse
from typing import Tuple, cast

import numpy as np
import tensorflow as tf

import flower as flwr

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT

tf.get_logger().setLevel("ERROR")

BATCH_SIZE = 32
SAMPLE_TRAIN = 150
SAMPLE_TEST = 50


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
        default=100,
        help="CIFAR version, allowed values: 10 or 100 (default: 100)",
    )
    args = parser.parse_args()
    print(f"Run client, cid {args.cid}, partition {args.partition}, CIFAR-{args.cifar}")

    # Load model and data
    model = load_model(input_shape=(32, 32, 3), num_classes=args.cifar)
    xy_train, xy_test = load_data(num_classes=args.cifar)

    # Start client
    client = CifarClient(args.cid, model, xy_train, xy_test)
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


class CifarClient(flwr.Client):
    """Flower client implementing CIAFR-10/100 image classification using TensorFlow."""

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

    def get_weights(self) -> flwr.Weights:
        print(f"[client:{self.cid}] get_weights")
        return cast(flwr.Weights, self.model.get_weights())

    def fit(self, weights: flwr.Weights) -> Tuple[flwr.Weights, int]:
        print(f"[client:{self.cid}] fit")
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Train the local model using the local dataset
        self.model.fit(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=1)
        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_train)

    def evaluate(self, weights: flwr.Weights) -> Tuple[int, float]:
        print(f"[client:{self.cid}] evaluate")
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        # Evaluate the updated model on the local dataset
        loss, _ = self.model.evaluate(self.x_test, self.y_test, batch_size=SAMPLE_TEST)
        # Return the number of evaluation examples and the evaltion result (loss)
        return len(self.x_test), float(loss)


def load_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Create a ResNet-50 (v2) instance"""
    model = tf.keras.applications.ResNet50V2(
        weights=None, include_top=True, input_shape=input_shape, classes=num_classes
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_data(
    num_classes: int, subtract_pixel_mean: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load, normalize, and sample CIFAR-10/100."""
    cifar = (
        tf.keras.datasets.cifar10 if num_classes == 10 else tf.keras.datasets.cifar100
    )
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

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

    # Take a random subset of the dataset to simulate having different local datasets
    idxs_train = np.random.choice(np.arange(len(x_train)), SAMPLE_TRAIN, replace=False)
    x_train_sample, y_train_sample = x_train[idxs_train], y_train[idxs_train]
    idxs_test = np.random.choice(np.arange(len(x_test)), SAMPLE_TEST, replace=False)
    x_test_sample, y_test_sample = x_test[idxs_test], y_test[idxs_test]

    return (x_train_sample, y_train_sample), (x_test_sample, y_test_sample)


if __name__ == "__main__":
    main()
