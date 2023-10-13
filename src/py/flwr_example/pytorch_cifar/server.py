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
"""Minimal example on how to start a simple Flower server."""


import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision

import flwr as fl

from . import DEFAULT_SERVER_ADDRESS, cifar

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
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
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Load evaluation data
    _, testset = cifar.load_data()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        evaluate_fn=get_evaluate_fn(testset),
        on_fit_config_fn=fit_config,
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(server_round),
        "epochs": str(1),
        "batch_size": str(32),
    }
    return config


def get_evaluate_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.NDArrays) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = cifar.load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return cifar.test(model, testloader, device=DEVICE)

    return evaluate


if __name__ == "__main__":
    main()
