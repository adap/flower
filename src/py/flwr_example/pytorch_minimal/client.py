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
"""Flower client example using PyTorch for CIFAR-10 image classification."""


import argparse
import timeit
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torchvision

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

from . import DEFAULT_SERVER_ADDRESS, cifar

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

# Flower Client
class CifarClient(fl.client.KerasClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_weights(self) -> Weights:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        # Set model weights from a list of NumPy ndarrays.
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, weights: Weights, config: Dict[str, str]) -> Tuple[Weights, int, int]:
        # Set model parameters
        self.set_weights(weights)

        # Train model
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)

        # Return the updated model parameters
        return self.get_weights(), len(self.trainloader), len(self.testloader)

    def evaluate(
        self, weights: Weights, config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Use provided weights to update the local model
        self.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.testloader), float(loss), float(accuracy)


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = cifar.Net()
    model.to(DEVICE)
    trainloader, testloader = cifar.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader)
    fl.client.start_keras_client(DEFAULT_SERVER_ADDRESS, client)


if __name__ == "__main__":
    main()
