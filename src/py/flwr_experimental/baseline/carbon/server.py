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
"""Flower server for Fashion-MNIST image classification."""


import argparse
import math
from logging import ERROR, INFO
from typing import Callable, Dict, Optional

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.common import get_eval_fn
from flwr_experimental.baseline.dataset import tf_fashion_mnist_partitioned
from flwr_experimental.baseline.model import orig_cnn
from flwr_experimental.baseline.tf_fashion_mnist.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, SEED


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--log_host", type=str, help="HTTP log handler host (no default)",
    # )
    parser.add_argument(
        "--rounds", type=int, help="FL rounds to run.",
    )
    parser.add_argument(
        "--sample_fraction", type=float, help=".",
    )
    parser.add_argument(
        "--min_sample_size", type=int, help=".",
    )
    parser.add_argument(
        "--min_num_clients", type=int, help=".",
    )

    return parser.parse_args()


def main() -> None:
    """Start server and train a number of rounds."""
    args = parse_args()

    # Load evaluation data
    (_, _), (x_test, y_test) = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=1.0, num_partitions=1
    )

    # Load model (for centralized evaluation)
    model = orig_cnn(input_shape=(28, 28, 1), seed=SEED)

    # Create client_manager
    client_manager = fl.server.SimpleClientManager()

    # Strategy
    eval_fn = get_eval_fn(model=model, num_classes=10, xy_test=(x_test, y_test))
    on_fit_config_fn = get_on_fit_config_fn(
        lr_initial=0.01, timeout=None, partial_updates=False,
    )

    # Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=eval_fn,
        on_fit_config_fn=on_fit_config_fn,
    )

    # Run server
    log(INFO, "Instantiating server, strategy: %s", str(strategy))
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS, server, config={"num_rounds": args.rounds},
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
