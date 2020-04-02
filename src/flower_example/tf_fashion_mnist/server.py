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
"""Minimal example on how to start a simple Flower server."""


import argparse
from typing import Callable, Optional, Tuple

import numpy as np

import flower as flwr

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT
from .client import get_lr_initial, load_data, load_model


def main() -> None:
    """Start server and train five rounds."""
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
    parser.add_argument("--cid", type=str, help="Client CID (no default)")
    args = parser.parse_args()

    # Load evaluation data
    _, xy_test = load_data(partition=0, num_clients=1)

    # Create client_manager, strategy, and server
    client_manager = flwr.SimpleClientManager()
    strategy = flwr.strategy.DefaultStrategy(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(xy_test=xy_test),
    )
    server = flwr.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    flwr.app.start_server(
        args.grpc_server_address,
        args.grpc_server_port,
        server,
        config={"num_rounds": args.rounds},
    )


def get_eval_fn(
    xy_test: Tuple[np.ndarray, np.ndarray]
) -> Callable[[flwr.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: flwr.Weights) -> Optional[Tuple[float, float]]:
        """Use entire Fashion-MNIST test set for evaluation."""
        model = load_model(learning_rate=get_lr_initial())
        model.set_weights(weights)
        loss, acc = model.evaluate(xy_test[0], xy_test[1], batch_size=len(xy_test))
        return float(loss), float(acc)

    return evaluate


if __name__ == "__main__":
    main()
