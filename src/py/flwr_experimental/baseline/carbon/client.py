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
from logging import ERROR, INFO

import tensorflow as tf

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.common import VisionClassificationClient
from flwr_experimental.baseline.dataset import tf_fashion_mnist_partitioned
from flwr_experimental.baseline.model import orig_cnn
from flwr_experimental.baseline.setting import ClientSetting
from flwr_experimental.baseline.tf_fashion_mnist.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, SEED

tf.get_logger().setLevel("ERROR")


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=str, required=True, help="Client cid.")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
    )
    # parser.add_argument(
    #     "--log_host", type=str, help="HTTP log handler host (no default)",
    # )
    parser.add_argument(
        "--num_clients", type=int, help="IID fraction.",
    )
    parser.add_argument(
        "--iid_fraction", type=float, help="IID fraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, create and start Fashion-MNIST client."""
    args = parse_args()

    # Load model
    model = orig_cnn(input_shape=(28, 28, 1), seed=SEED)

    # Load local data partition
    (
        (xy_train_partitions, xy_test_partitions),
        _,
    ) = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=args.iid_fraction, num_partitions=args.num_clients,
    )
    x_train, y_train = xy_train_partitions[int(args.cid)]
    x_test, y_test = xy_test_partitions[int(args.cid)]

    # Start client
    client = VisionClassificationClient(
        args.cid,
        model,
        (x_train, y_train),
        (x_test, y_test),
        delay_factor=0.0,
        num_classes=10,
        augment=False,
        augment_horizontal_flip=False,
        augment_offset=1,
    )
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
