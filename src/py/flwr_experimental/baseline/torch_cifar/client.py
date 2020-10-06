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
import timeit
from logging import ERROR, INFO

import torch
import torchvision
from torchvision import transforms

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from flwr.common.logger import configure, log
from flwr_experimental.baseline.setting import ClientSetting
from flwr_experimental.baseline.torch_cifar.settings import SETTINGS, get_setting

from . import DEFAULT_SERVER_ADDRESS, cifar

# pylint: disable=no-member
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        model: torch.nn.ModuleList,
        trainset: torch.utils.data.Dataset,
        testset: torch.utils.data.Dataset,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def get_parameters(self) -> ParametersRes:
        log(INFO, "Client %s: get_parameters", self.cid)
        weights: Weights = cifar.get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, "Client %s: fit", self.cid)
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config

        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        cifar.set_weights(self.model, weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        cifar.train(
            model=self.model,
            trainloader=trainloader,
            epochs=epochs,
            device=DEVICE,
            # batches_per_episode=5,
        )

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = cifar.get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, "Client %s: evaluate", self.cid)
        weights = fl.common.parameters_to_weights(ins.parameters)
        _ = ins.config

        # Use provided weights to update the local model
        cifar.set_weights(self.model, weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = cifar.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=SETTINGS.keys(),
        help="Setting identifier (key).",
    )
    parser.add_argument(
        "--cid",
        type=str,
        required=True,
        help="Client ID.",
    )
    return parser.parse_args()


class ClientSettingNotFound(Exception):
    """Raise when client setting could not be found."""


def get_client_setting(setting: str, cid: str) -> ClientSetting:
    """Return client setting based on setting name and cid."""
    for client_setting in get_setting(setting).clients:
        if client_setting.cid == cid:
            return client_setting

    raise ClientSettingNotFound()


def main() -> None:
    """Load data, create and start CIFAR-10/100 client."""
    args = parse_args()

    client_setting = get_client_setting(args.setting, args.cid)

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)
    log(INFO, "Starting client, settings: %s", client_setting)

    # Load model
    model = cifar.load_model(DEVICE)

    # Load local data partition
    trainset, testset = cifar.load_data(
        cid=int(client_setting.cid), root_dir=cifar.DATA_ROOT, load_testset=True
    )

    # Start client
    client = CifarClient(
        cid=client_setting.cid,
        model=model,
        trainset=trainset,
        testset=testset,
    )
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)
        # Raise the error again so the exit code is correct
        raise err
