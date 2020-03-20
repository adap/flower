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
"""Example on how to start a simple Flower server."""
import argparse
from typing import Tuple

import flower as flwr

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT


class CifarStrategy(flwr.Strategy):
    """Strategy using at least three clients for training and evaluation."""

    def __init__(
        self, sample_fraction: float, min_sample_size: int, min_num_clients: int
    ) -> None:
        super().__init__()
        self.sample_fraction = sample_fraction
        self.min_sample_size = min_sample_size
        self.min_num_clients = min_num_clients

    def should_evaluate(self) -> bool:
        """Evaluate every round."""
        return True

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine number of clients used for training."""
        sample_size = int(
            max(num_available_clients * self.sample_fraction, self.min_sample_size)
        )
        return sample_size, self.min_num_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine number of clients used for evaluation."""
        sample_size = int(
            max(num_available_clients * self.sample_fraction, self.min_sample_size)
        )
        return sample_size, self.min_num_clients


def main() -> None:
    """Start server and train for three rounds."""
    parser = argparse.ArgumentParser(description="Flower/TensorFlower")
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

    client_manager = flwr.SimpleClientManager()
    strategy = CifarStrategy(
        args.sample_fraction, args.min_sample_size, args.min_num_clients
    )
    server = flwr.Server(client_manager=client_manager, strategy=strategy)
    flwr.app.start_server(
        args.grpc_server_address,
        args.grpc_server_port,
        server,
        config={"num_rounds": args.rounds},
    )


if __name__ == "__main__":
    main()
