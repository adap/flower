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
"""Flower server example."""


import argparse
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision

import flwr as fl

from . import cifar


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights


def main() -> None:
    """Start server and train five rounds."""
    # Load evaluation data
    _, testloader = cifar.load_data()

    # Create client_manager, strategy, and server
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        eval_fn=get_eval_fn(testloader),
        on_fit_config_fn=fit_config,
    )

    # Run server
    fl.server.start_server(
        config={"num_rounds": 3},
        strategy=strategy,
    )


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(32),
    }
    return config


def get_eval_fn(
    testloader: torch.utils.data.DataLoader,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    # pylint: disable=no-member
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # pylint: enable=no-member

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = cifar.Net()
        model.set_weights(weights)
        model.to(DEVICE)
        return cifar.test(model, testloader, device=DEVICE)

    return evaluate


if __name__ == "__main__":
    main()
