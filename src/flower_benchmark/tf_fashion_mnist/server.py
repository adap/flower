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
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import flower as flwr

from . import (
    DEFAULT_GRPC_SERVER_ADDRESS,
    DEFAULT_GRPC_SERVER_PORT,
    client,
    fashion_mnist,
)


def main() -> None:
    """Start server and train `--rounds` number of rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (IPv6, default: [::])",
    )
    parser.add_argument(
        "--grpc_server_port",
        type=int,
        default=DEFAULT_GRPC_SERVER_PORT,
        help="gRPC server port (default: 8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of available clients used for fit/evaluate (default: 0.1)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=1,
        help="Minimum number of clients used for fit/evaluate (default: 1)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=1,
        help="Minimum number of available clients required for sampling (default: 1)",
    )
    parser.add_argument(
        "--lr_initial",
        type=float,
        default=0.1,
        help="Initial learning rate (default: 0.1)",
    )
    parser.add_argument("--cid", type=str, help="Client CID (no default)")
    parser.add_argument(
        "--dry_run", type=bool, default=False, help="Dry run (default: False)"
    )
    args = parser.parse_args()

    # Load evaluation data
    _, xy_test = client.load_data(partition=0, num_clients=1, dry_run=args.dry_run)

    # Create client_manager, strategy, and server
    client_manager = flwr.SimpleClientManager()
    strategy = flwr.strategy.DefaultStrategy(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(num_classes=10, xy_test=xy_test),
        on_fit_config_fn=get_on_fit_config_fn(args.lr_initial),
    )
    server = flwr.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    flwr.app.start_server(
        args.grpc_server_address,
        args.grpc_server_port,
        server,
        config={"num_rounds": args.rounds},
    )


def get_on_fit_config_fn(lr_initial: float) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(1),
            "batch_size": str(64),
            "lr_initial": str(lr_initial),
            "lr_decay": str(0.99),
            "timeout": str(20),
            "partial_updates": "1",
        }
        return config

    return fit_config


def get_eval_fn(
    num_classes: int, xy_test: Tuple[np.ndarray, np.ndarray]
) -> Callable[[flwr.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""
    ds_test = fashion_mnist.build_dataset(
        xy_test[0],
        xy_test[1],
        num_classes=num_classes,
        shuffle_buffer_size=0,
        augment=False,
    )

    def evaluate(weights: flwr.Weights) -> Optional[Tuple[float, float]]:
        """Use entire Fashion-MNIST test set for evaluation."""
        model = client.load_model(input_shape=(28, 28, 1))
        model.set_weights(weights)
        loss, acc = fashion_mnist.keras_evaluate(
            model, ds_test, batch_size=len(xy_test[0])
        )
        return loss, acc

    return evaluate


if __name__ == "__main__":
    main()
