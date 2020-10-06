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
"""Flower client using PyTorch for CIFAR-10 image classification."""


import argparse
from logging import ERROR, INFO
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision
from torchvision import transforms

import flwr as fl
from flwr.common.logger import configure, log
from flwr_experimental.baseline.torch_cifar.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, cifar

# pylint: disable=no-member
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


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
    if server_setting.strategy != "fedavg":
        msg = f"Configured strategy `{server_setting.strategy}`"
        msg += ", but only `fedavg` is supported."
        raise Exception(msg)

    # Load model (for centralized evaluation)
    model = cifar.load_model(DEVICE)

    # Load evaluation data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=cifar.DATA_ROOT,
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )

    # Create client_manager
    client_manager = fl.server.SimpleClientManager()

    # Strategy
    eval_fn = get_eval_fn(model=model, testset=testset)
    fit_config_fn = get_on_fit_config_fn(lr_initial=server_setting.lr_initial)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=server_setting.sample_fraction,
        min_fit_clients=server_setting.min_sample_size,
        min_available_clients=server_setting.min_num_clients,
        eval_fn=eval_fn,
        on_fit_config_fn=fit_config_fn,
    )

    # Run server
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        server,
        config={"num_rounds": server_setting.rounds},
    )


def get_eval_fn(
    model: torch.nn.ModuleList, testset: torchvision.datasets.CIFAR10
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use entire test set for evaluation."""

        # Use provided weights to update the local model
        cifar.set_weights(model, weights)

        # Evaluate the updated model on the local dataset
        loss, accuracy = cifar.test(model, testloader, device=DEVICE)
        return loss, accuracy

    return evaluate


def get_on_fit_config_fn(lr_initial: float) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(1),
            "batch_size": str(16),
            "lr_initial": str(lr_initial),
            "lr_decay": str(0.99),
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
