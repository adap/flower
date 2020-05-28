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


import argparse
from typing import Callable, Dict, Optional, Tuple

import flower as fl

from . import DEFAULT_SERVER_ADDRESS


def main() -> None:
    # Create ClientManager & Strategy
    client_manager = fl.SimpleClientManager()
    strategy = fl.strategy.DefaultStrategy(
        fraction_fit=0.1,
        min_fit_clients=2,
        min_available_clients=2,
        eval_fn=centralized_evaluation_function,
        on_fit_config_fn=get_on_fit_config_fn(0.01, 60),
    )

    # Run server
    server = fl.Server(client_manager=client_manager, strategy=strategy)
    fl.app.start_server(
        DEFAULT_SERVER_ADDRESS, server, config={"num_rounds": 10},
    )


def centralized_evaluation_function(
    weights: fl.Weights,
) -> Optional[Tuple[float, float]]:
    """Use entire test set for evaluation."""
    # TODO Evaluate weights and return
    lss, acc = 0.1, 0.1

    return lss, acc


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
