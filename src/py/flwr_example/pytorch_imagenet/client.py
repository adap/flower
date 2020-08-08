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
"""Flower client example using PyTorch for Imagenet image classification."""

from collections import OrderedDict

import argparse
import timeit

import torch
import torchvision

import flwr as fl

import numpy as np

import imagenet
import torchvision.models as models

DEFAULT_SERVER_ADDRESS = "[::]:8080"

# pylint: disable=no-member
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

def get_weights(model) -> fl.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights: fl.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {k: torch.Tensor(np.atleast_1d(v)) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)

class ImageNetClient(fl.Client):
    """Flower client implementing ImageNet image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        nb_clients: int
    ) -> None:
        super().__init__(cid)
        self.model = models.resnet18().to(DEVICE)
        self.trainset = trainset
        self.testset = testset
        self.nb_clients = nb_clients

    def get_parameters(self) -> fl.ParametersRes:
        print(f"Client {self.cid}: get_parameters")
        weights: fl.Weights = get_weights(self.model)
        parameters = fl.weights_to_parameters(weights)
        return fl.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.FitIns) -> fl.FitRes:

        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        print(f"Client {self.cid}: fit")

        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config = ins[1]
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        set_weights(self.model, weights)

        # Get the data corresponding to this client
        dataset_size = len(self.trainset)
        nb_samples_per_clients = dataset_size // self.nb_clients
        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)

        # Get starting and ending indices w.r.t cid
        start_ind = int(self.cid) * nb_samples_per_clients
        end_ind = (int(self.cid) * nb_samples_per_clients) + nb_samples_per_clients
        train_sampler = torch.utils.data.SubsetRandomSampler(dataset_indices[start_ind:end_ind])

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler
        )

        imagenet.train(self.model, trainloader, epochs=epochs, device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.Weights = get_weights(self.model)
        params_prime = fl.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return params_prime, num_examples_train, num_examples_train, fit_duration

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:

        # Set the set so we are sure to generate the same batches
        # accross all clients.
        np.random.seed(123)

        print(f"Client {self.cid}: evaluate")

        config = ins[1]
        batch_size = int(config["batch_size"])

        weights = fl.parameters_to_weights(ins[0])

        # Use provided weights to update the local model
        set_weights(self.model, weights)

        # Get the data corresponding to this client
        dataset_size = len(self.test_set)
        nb_samples_per_clients = dataset_size // self.nb_clients
        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)

        # Get starting and ending indices w.r.t cid
        start_ind = int(self.cid) * nb_samples_per_clients
        end_ind = (int(self.cid) * nb_samples_per_clients) + nb_samples_per_clients
        test_sampler = torch.utils.data.SubsetRandomSampler(dataset_indices[start_ind:end_ind])

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, sampler=test_sampler
        )

        loss, accuracy = imagenet.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(self.testset), float(loss), float(accuracy)


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="ImageNet datapath"
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    parser.add_argument(
        "--nb_clients", type=int, default=40, help="Total number of clients",
    )
    args = parser.parse_args()

    # Configure logger
    fl.logger.configure(f"client_{args.cid}", host=args.log_host)


    trainset, testset = imagenet.load_data(args.data_path)

    # Start client
    client = ImageNetClient(args.cid, trainset, testset, args.nb_clients)
    fl.app.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
