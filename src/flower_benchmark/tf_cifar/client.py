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
from logging import ERROR

import tensorflow as tf

import flower as flwr
from flower.logger import configure, log
from flower_benchmark.common import VisionClassificationClient
from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.model import resnet50v2
from flower_benchmark.setting import ClientSetting
from flower_benchmark.tf_fashion_mnist.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, NUM_CLASSES, SEED

tf.get_logger().setLevel("ERROR")


class ClientSettingNotFound(Exception):
    """Raise when client setting could not be found."""


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )
    parser.add_argument("--cid", type=str, required=True, help="Client cid.")
    return parser.parse_args()


def get_client_setting(setting: str, cid: str) -> ClientSetting:
    """Return client setting based on setting name and cid."""
    for client_setting in get_setting(setting).clients:
        if client_setting.cid == cid:
            return client_setting

    raise ClientSettingNotFound()


def main() -> None:
    """Load data, create and start CIFAR-10/100 client."""
    args = parse_args()

    client_setting = get_setting(args.setting).clients[args.index]

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)

    # Load model
    model = resnet50v2(input_shape=(32, 32, 3), num_classes=NUM_CLASSES, seed=SEED)

    # Load local data partition
    (xy_train_partitions, xy_test_partitions), _ = tf_cifar_partitioned.load_data(
        iid_fraction=client_setting.iid_fraction,
        num_partitions=client_setting.num_clients,
        cifar100=False,
    )
    xy_train = xy_train_partitions[client_setting.partitions]
    xy_test = xy_test_partitions[client_setting.partitions]
    if client_setting.dry_run:
        xy_train = xy_train[0:100]
        xy_test = xy_test[0:50]

    # Start client
    client = VisionClassificationClient(
        client_setting.cid,
        model,
        xy_train,
        xy_test,
        client_setting.delay_factor,
        NUM_CLASSES,
    )
    flwr.app.start_client(args.server_address, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
