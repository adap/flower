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
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors


from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import flower as fl

DATA_ROOT = "./data/cifar-10"
PATH = "./cifar_net.pth"


# pylint: disable-msg=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = cast(Tensor, self.pool(F.relu(self.conv1(x))))
        x = cast(Tensor, self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def load_model() -> Net:
    """Load a simple CNN."""
    return Net()
