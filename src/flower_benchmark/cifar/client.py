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
"""Baseline client."""


import argparse
import timeit
from logging import ERROR, INFO
from typing import Tuple

import numpy as np
import torch

import flower as fl
from flower.logger import configure, log
from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.setting import ClientSetting

from . import DEFAULT_SERVER_ADDRESS, cifar
from .settings import SETTINGS, get_setting

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


class ClientSettingNotFound(Exception):
    """Raise when client setting could not be found."""


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
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )
    parser.add_argument("--cid", type=str, required=True, help="Client cid.")
    return parser.parse_args()


def get_client_setting(setting: str, cid: str) -> ClientSetting:
    """Return client setting based on setting name and cid."""
    for client_setting in get_setting(setting).clients:
        if client_setting.cid == cid:
            return client_setting

    raise ClientSettingNotFound()


class CifarClient(fl.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cid: str,
        model: cifar.Net,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        delay_factor: float,
    ) -> None:
        super().__init__(cid)
        self.model = model
        self.trainset = cifar.ds_from_nda(xy_train[0], xy_train[1])
        self.testset = cifar.ds_from_nda(xy_test[0], xy_test[1])
        self.delay_factor = delay_factor

    def get_parameters(self) -> fl.ParametersRes:
        log(INFO, "Client %s: get_parameters", self.cid)
        weights: fl.Weights = self.model.get_weights()
        parameters = fl.weights_to_parameters(weights)
        return fl.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.FitIns) -> fl.FitRes:
        log(INFO, "Client %s: fit", self.cid)

        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config = ins[1]
        fit_begin = timeit.default_timer()

        # TODO: augment (flip, color, crop)  # pylint: disable=fixme
        # TODO: delay factor  # pylint: disable=fixme

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        cifar.train(self.model, trainloader, epochs=epochs, device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.Weights = self.model.get_weights()
        params_prime = fl.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return params_prime, num_examples_train, num_examples_train, fit_duration

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:
        log(INFO, "Client %s: evaluate", self.cid)

        weights = fl.parameters_to_weights(ins[0])

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = cifar.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.testset), float(loss), float(accuracy)


def main() -> None:
    """Load data, create and start client."""
    args = parse_args()

    client_setting = get_client_setting(args.setting, args.cid)

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)
    log(INFO, "Starting client, settings: %s", client_setting)

    # Load model
    model = cifar.load_model()

    # Load local data partition
    ((xy_train_partitions, xy_test_partitions), _,) = tf_cifar_partitioned.load_data(
        iid_fraction=client_setting.iid_fraction,
        num_partitions=client_setting.num_clients,
    )
    x_train, y_train = xy_train_partitions[client_setting.partition]
    x_test, y_test = xy_test_partitions[client_setting.partition]
    if client_setting.dry_run:
        x_train = x_train[0:100]
        y_train = y_train[0:100]
        x_test = x_test[0:50]
        y_test = y_test[0:50]

    # Start client
    client = CifarClient(
        cid=client_setting.cid,
        model=model,
        xy_train=(x_train, y_train),
        xy_test=(x_test, y_test),
        delay_factor=client_setting.delay_factor,
    )
    fl.app.start_client(args.server_address, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
