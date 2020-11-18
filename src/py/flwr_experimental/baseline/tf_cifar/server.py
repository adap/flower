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
import math
from logging import ERROR, INFO
from typing import Callable, Dict, Optional

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.common import get_eval_fn
from flwr_experimental.baseline.dataset import tf_cifar_partitioned
from flwr_experimental.baseline.model import resnet50v2
from flwr_experimental.baseline.tf_cifar.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, NUM_CLASSES, SEED


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--log_host",
        type=str,
        help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=SETTINGS.keys(),
        help="Setting to run.",
    )

    return parser.parse_args()


def main() -> None:
    """Start server and train a number of rounds."""
    args = parse_args()

    # Configure logger
    configure(identifier="server", host=args.log_host)

    server_setting = get_setting(args.setting).server
    log(INFO, "server_setting: %s", server_setting)

    # Load evaluation data
    (_, _), (x_test, y_test) = tf_cifar_partitioned.load_data(
        iid_fraction=0.0, num_partitions=1, cifar100=NUM_CLASSES == 100
    )
    if server_setting.dry_run:
        x_test = x_test[0:50]
        y_test = y_test[0:50]

    # Load model (for centralized evaluation)
    model = resnet50v2(input_shape=(32, 32, 3), num_classes=NUM_CLASSES, seed=SEED)

    # Create client_manager
    client_manager = fl.server.SimpleClientManager()

    # Strategy
    eval_fn = get_eval_fn(
        model=model, num_classes=NUM_CLASSES, xy_test=(x_test, y_test)
    )
    fit_config_fn = get_on_fit_config_fn(
        lr_initial=server_setting.lr_initial,
        timeout=server_setting.training_round_timeout,
        partial_updates=server_setting.partial_updates,
    )

    if server_setting.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
        )

    if server_setting.strategy == "fast-and-slow":
        if server_setting.training_round_timeout is None:
            raise ValueError(
                "No `training_round_timeout` set for `fast-and-slow` strategy"
            )
        strategy = fl.server.strategy.FastAndSlow(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            importance_sampling=server_setting.importance_sampling,
            dynamic_timeout=server_setting.dynamic_timeout,
            dynamic_timeout_percentile=0.8,
            alternating_timeout=server_setting.alternating_timeout,
            r_fast=1,
            r_slow=1,
            t_fast=math.ceil(0.5 * server_setting.training_round_timeout),
            t_slow=server_setting.training_round_timeout,
        )

    # Run server
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        server,
        config={"num_rounds": server_setting.rounds},
    )


def get_on_fit_config_fn(
    lr_initial: float, timeout: Optional[int], partial_updates: bool
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(1),
            "batch_size": str(32),
            "lr_initial": str(lr_initial),
            "lr_decay": str(0.99),
            "partial_updates": "1" if partial_updates else "0",
        }
        if timeout is not None:
            config["timeout"] = str(timeout)

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
