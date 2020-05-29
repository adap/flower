# Copyright 2020 The Flower Authors. All Rights Reserved.
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
import argparse
from logging import ERROR
from typing import Callable, Dict

import flower as fl


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
    parser = argparse.ArgumentParser(description="FlowerSpeechBrain")
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
    args = parser.parse_args()
    # Create ClientManager & Strategy
    client_manager = fl.SimpleClientManager()
    strategy = fl.strategy.DefaultStrategy(
        fraction_fit=0.1,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config_fn(0.01, 60),
    )
    # Run server
    server = fl.Server(client_manager=client_manager, strategy=strategy)
    fl.app.start_server(
        args.grpc_server_address,
        args.grpc_server_port,
        server,
        config={"num_rounds": 10},
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
            "timeout": str(timeout),
        }
        return config

    return fit_config


if __name__ == "__main__":
    main()
