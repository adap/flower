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
"""Flower client using TensorFlow for Spoken Keyword classification."""


import argparse
from logging import ERROR

import tensorflow as tf

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.common import VisionClassificationClient
from flwr_experimental.baseline.dataset import tf_hotkey_partitioned
from flwr_experimental.baseline.model import keyword_cnn
from flwr_experimental.baseline.setting import ClientSetting
from flwr_experimental.baseline.tf_hotkey.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, SEED

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
        help=f"Server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
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
    """Load data, create and start client."""
    args = parse_args()

    client_setting = get_client_setting(args.setting, args.cid)

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)

    # Load model
    model = keyword_cnn(input_shape=(80, 40, 1), seed=SEED)

    # Load local data partition
    ((xy_train_partitions, xy_test_partitions), _,) = tf_hotkey_partitioned.load_data(
        iid_fraction=client_setting.iid_fraction,
        num_partitions=client_setting.num_clients,
    )
    (x_train, y_train) = xy_train_partitions[client_setting.partition]
    (x_test, y_test) = xy_test_partitions[client_setting.partition]
    if client_setting.dry_run:
        x_train = x_train[0:100]
        y_train = y_train[0:100]
        x_test = x_test[0:50]
        y_test = y_test[0:50]

    # Start client
    client = VisionClassificationClient(
        client_setting.cid,
        model,
        (x_train, y_train),
        (x_test, y_test),
        client_setting.delay_factor,
        10,
        normalization_factor=100.0,
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
