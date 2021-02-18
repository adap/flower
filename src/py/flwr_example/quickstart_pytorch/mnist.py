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

"""PyTorch MNIST image classification.

The code is generally adapted from PyTorch's Basic MNIST Example. 
The original code can be inspected in the official PyTorch github:

https://github.com/pytorch/examples/blob/master/mnist/main.py
"""


# mypy: ignore-errors
# pylint: disable=W0223

import timeit
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms

import flwr as fl


def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
) -> torch.utils.data.DataLoader:
    """Helper function to partition datasets

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to be partitioned into *number_of_clients* subsets.

    batch_size: int
        Size of mini-batches used by the returned DataLoader.

    client_id: int
        Unique integer used for selecting a specific partition.

    number_of_clients: int
        Total number of clients launched during training. This value dictates the number of partitions to be created.


    Returns
    -------
    data_loader: torch.utils.data.Dataset
        DataLoader for specific client_id considering number_of_clients partitions.

    """

    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t CLIENT_ID
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler
    )
    return data_loader


def load_data(
    data_root: str,
    train_batch_size: int,
    test_batch_size: int,
    cid: int,
    nb_clients: int,
) -> Tuple[DataLoader, DataLoader]:
    """Helper function that loads both training and test datasets for MNIST.

    Parameters
    ----------
    data_root: str
        Directory where MNIST dataset will be stored.

    train_batch_size: int
        Mini-batch size for training set.

    test_batch_size: int
        Mini-batch size for test set.

    cid: int
        Client ID used to select a specific partition.

    nb_clients: int
        Total number of clients launched during training. This value dictates the number of unique to be created.


    Returns
    -------
    (train_loader, test_loader): Tuple[DataLoader, DataLoader]
        Tuple contaning DataLoaders for training and test sets.

    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    # Create partitioned datasets based on the total number of clients and client_id
    train_loader = dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    test_loader = dataset_partitioner(
        dataset=test_dataset,
        batch_size=test_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    return (train_loader, test_loader)


class MNISTNet(nn.Module):
    """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""

    def __init__(self) -> None:
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x: Tensor
            Mini-batch of shape (N,28,28) containing images from MNIST dataset.


        Returns
        -------
        output: Tensor
            The probability density of the output being from a specific class given the input.

        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Train routine based on 'Basic MNIST Example'

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model used in this example.

    train_loader: torch.utils.data.DataLoader
        DataLoader used in training.

    epochs: int
        Number of epochs to run in each round.

    device: torch.device
         (Default value = torch.device("cpu"))
         Device where the network will be trained within a client.

    Returns
    -------
    num_examples_train: int
        Number of total samples used during training.

    """
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print(f"Training {epochs} epoch(s) w/ {len(train_loader)} mini-batches each")
    for epoch in range(epochs):  # loop over the dataset multiple times
        print()
        loss_epoch: float = 0.0
        num_examples_train: int = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Grab mini-batch and transfer to device
            data, target = data.to(device), target.to(device)
            num_examples_train += len(data)

            # Zero gradients
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            if batch_idx % 10 == 8:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\t\t\t\t".format(
                        epoch,
                        num_examples_train,
                        len(train_loader) * train_loader.batch_size,
                        100.0
                        * num_examples_train
                        / len(train_loader)
                        / train_loader.batch_size,
                        loss.item(),
                    ),
                    end="\r",
                    flush=True,
                )
        scheduler.step()
    return num_examples_train


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, float]:
    """Test routine 'Basic MNIST Example'

    Parameters
    ----------
    model: torch.nn.Module :
        Neural network model used in this example.

    test_loader: torch.utils.data.DataLoader :
        DataLoader used in test.

    device: torch.device :
         (Default value = torch.device("cpu"))
         Device where the network will be tested within a client.

    Returns
    -------
        Tuple containing the total number of test samples, the test_loss, and the accuracy evaluated on the test set.

    """
    model.eval()
    test_loss: float = 0
    correct: int = 0
    num_test_samples: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_test_samples += len(data)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_test_samples

    return (num_test_samples, test_loss, correct / num_test_samples)


class PytorchMNISTClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""

    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = MNISTNet().to(device)
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays.

        Parameters
        ----------
        weights: fl.common.Weights
            Weights received by the server and set to local model


        Returns
        -------

        """
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> fl.common.ParametersRes:
        """Encapsulates the weights into Flower Parameters """
        weights: fl.common.Weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Trains the model on local dataset

        Parameters
        ----------
        ins: fl.common.FitIns
           Parameters sent by the server to be used during training.

        Returns
        -------
            Set of variables containing the new set of weights and information the client.

        """

        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        fit_begin = timeit.default_timer()

        # Set model parameters/weights
        self.set_weights(weights)

        # Train model
        num_examples_train: int = train(
            self.model, self.train_loader, epochs=self.epochs, device=self.device
        )

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.Weights = self.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """

        Parameters
        ----------
        ins: fl.common.EvaluateIns
           Parameters sent by the server to be used during testing.


        Returns
        -------
            Information the clients testing results.

        """
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.set_weights(weights)

        (
            num_examples_test,
            test_loss,
            accuracy,
        ) = test(self.model, self.test_loader, device=self.device)
        print(
            f"Client {self.cid} - Evaluate on {num_examples_test} samples: Average loss: {test_loss:.4f}, Accuracy: {100*accuracy:.2f}%\n"
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return fl.common.EvaluateRes(
            loss=float(test_loss),
            num_examples=num_examples_test,
            accuracy=float(accuracy),
        )
