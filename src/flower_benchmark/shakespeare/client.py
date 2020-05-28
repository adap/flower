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
from typing import Dict

import flower as fl

from . import DEFAULT_SERVER_ADDRESS


class ShakespeareClient(fl.Client):
    def __init__(self, cid: str):
        super().__init__(cid)

    def get_parameters(self) -> fl.ParametersRes:
        weights: fl.Weights = []  # TODO get current weights from local model
        parameters = fl.weights_to_parameters(weights)
        return fl.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.FitIns) -> fl.FitRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config: Dict[str, str] = ins[1]

        # Read training configuration
        epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        timeout = int(config["timeout"])

        # TODO update local model with provided weights
        # TODO train model on local dataset
        # TODO return tuple (trained parameters, training examples, training examples ceil, fit duration)

        return self.get_parameters(), num_examples, num_examples_ceil, fit_duration

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config: Dict[str, str] = ins[1]

        # TODO update local model with provided weights
        # TODO evaluate model on local dataset
        # TODO return tuple (number of local evaluation examples, loss, accuracy)

        return num_examples_test, loss, accuracy


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"Server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument("--cid", type=str, required=True, help="Client cid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logger
    fl.logger.configure(identifier=f"client:{args.cid}")

    # Start client
    client = ShakespeareClient(args.cid)
    fl.app.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
