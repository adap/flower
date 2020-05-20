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
"""Flower server for CIFAR-10/100 image classification."""


import argparse
from logging import ERROR
from typing import Callable, Dict

import flower as flwr
from flower.logger import configure, log
from flower_benchmark.common import get_eval_fn, load_partition
from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.model import resnet50v2
from flower_benchmark.tf_fashion_mnist.settings import SETTINGS, get_setting

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, NUM_CLASSES, SEED


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )

    return parser.parse_args()


def main() -> None:
    """Start server and train a number of rounds."""
    args = parse_args()

    # Configure logger
    configure(identifier="server", host=args.log_host)

    server_setting = get_setting(args.setting).server

    # Load evaluation data
    xy_partitions, xy_test = tf_cifar_partitioned.load_data(
        iid_fraction=0.0, num_partitions=1, cifar100=NUM_CLASSES == 100
    )
    _, xy_test = load_partition(
        xy_partitions,
        xy_test,
        partition=0,
        num_clients=1,
        seed=SEED,
        dry_run=server_setting.dry_run,
    )

    # Load model (for centralized evaluation)
    model = resnet50v2(input_shape=(32, 32, 3), num_classes=NUM_CLASSES, seed=SEED)

    # Create client_manager, strategy, and server
    client_manager = flwr.SimpleClientManager()
    strategy = flwr.strategy.DefaultStrategy(
        fraction_fit=server_setting.sample_fraction,
        min_fit_clients=server_setting.min_sample_size,
        min_available_clients=server_setting.min_num_clients,
        eval_fn=get_eval_fn(model=model, num_classes=NUM_CLASSES, xy_test=xy_test),
        on_fit_config_fn=get_on_fit_config_fn(
            server_setting.lr_initial, server_setting.training_round_timeout
        ),
    )
    # strategy = flwr.strategy.FastAndSlow(
    #     fraction_fit=args.sample_fraction,
    #     min_fit_clients=args.min_sample_size,
    #     min_available_clients=args.min_num_clients,
    #     eval_fn=get_eval_fn(model=model, num_classes=NUM_CLASSES, xy_test=xy_test),
    #     on_fit_config_fn=get_on_fit_config_fn(
    #         args.lr_initial, args.training_round_timeout
    #     ),
    #     r_fast=1,
    #     r_slow=1,
    #     t_fast=20,
    #     t_slow=40,
    # )

    server = flwr.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    flwr.app.start_server(
        DEFAULT_GRPC_SERVER_ADDRESS,
        DEFAULT_GRPC_SERVER_PORT,
        server,
        config={"num_rounds": server_setting.rounds},
    )


def get_on_fit_config_fn(
    lr_initial: float, timeout: int
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(1),
            "batch_size": str(64),
            "lr_initial": str(lr_initial),
            "lr_decay": str(0.99),
            "timeout": str(timeout),
            "partial_updates": "1",
        }
        return config

    return fit_config


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
