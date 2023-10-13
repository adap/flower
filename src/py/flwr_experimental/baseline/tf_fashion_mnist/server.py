# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower server for Fashion-MNIST image classification."""


import argparse
import math
from logging import ERROR, INFO
from typing import Callable, Dict, Optional

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.common import get_evaluate_fn
from flwr_experimental.baseline.dataset import tf_fashion_mnist_partitioned
from flwr_experimental.baseline.model import orig_cnn
from flwr_experimental.baseline.tf_fashion_mnist.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, SEED


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
    (_, _), (x_test, y_test) = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=0.0, num_partitions=1
    )
    if server_setting.dry_run:
        x_test = x_test[0:50]
        y_test = y_test[0:50]

    # Load model (for centralized evaluation)
    model = orig_cnn(input_shape=(28, 28, 1), seed=SEED)

    # Strategy
    evaluate_fn = get_evaluate_fn(model=model, num_classes=10, xy_test=(x_test, y_test))
    on_fit_config_fn = get_on_fit_config_fn(
        lr_initial=server_setting.lr_initial,
        timeout=server_setting.training_round_timeout,
        partial_updates=server_setting.partial_updates,
    )

    if server_setting.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
        )

    if server_setting.strategy == "fast-and-slow":
        if server_setting.training_round_timeout is None:
            raise ValueError(
                "No `training_round_timeout` set for `fast-and-slow` strategy"
            )
        t_fast = (
            math.ceil(0.5 * server_setting.training_round_timeout)
            if server_setting.training_round_timeout_short is None
            else server_setting.training_round_timeout_short
        )
        strategy = fl.server.strategy.FastAndSlow(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            importance_sampling=server_setting.importance_sampling,
            dynamic_timeout=server_setting.dynamic_timeout,
            dynamic_timeout_percentile=0.8,
            alternating_timeout=server_setting.alternating_timeout,
            r_fast=1,
            r_slow=1,
            t_fast=t_fast,
            t_slow=server_setting.training_round_timeout,
        )

    if server_setting.strategy == "fedfs-v0":
        if server_setting.training_round_timeout is None:
            raise ValueError("No `training_round_timeout` set for `fedfs-v0` strategy")
        t_fast = (
            math.ceil(0.5 * server_setting.training_round_timeout)
            if server_setting.training_round_timeout_short is None
            else server_setting.training_round_timeout_short
        )
        strategy = fl.server.strategy.FedFSv0(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            r_fast=1,
            r_slow=1,
            t_fast=t_fast,
            t_slow=server_setting.training_round_timeout,
        )

    if server_setting.strategy == "fedfs-v1":
        if server_setting.training_round_timeout is None:
            raise ValueError("No `training_round_timeout` set for `fedfs-v1` strategy")
        strategy = fl.server.strategy.FedFSv1(
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            dynamic_timeout_percentile=0.8,
            r_fast=1,
            r_slow=1,
            t_max=server_setting.training_round_timeout,
            use_past_contributions=True,
        )

    if server_setting.strategy == "qffedavg":
        strategy = fl.server.strategy.QFedAvg(
            q_param=0.2,
            qffl_learning_rate=0.1,
            fraction_fit=server_setting.sample_fraction,
            min_fit_clients=server_setting.min_sample_size,
            min_available_clients=server_setting.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
        )

    # Run server
    log(INFO, "Instantiating server, strategy: %s", str(strategy))
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        config={"num_rounds": server_setting.rounds},
        strategy=strategy,
    )


def get_on_fit_config_fn(
    lr_initial: float, timeout: Optional[int], partial_updates: bool
) -> Callable[[int], Dict[str, fl.common.Scalar]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, fl.common.Scalar] = {
            "epoch_global": str(server_round),
            "epochs": str(5),
            "batch_size": str(10),
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
