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
"""Flower server tests"""


from typing import List, Tuple

from .client import Client
from .server import eval_clients, fit_clients
from .typing import Weights


class SuccessClient(Client):
    """Test class."""

    def get_weights(self) -> Weights:
        # This method is not expected to be called
        raise Exception()

    def fit(self, weights: Weights) -> Tuple[Weights, int]:
        return [], 1

    def evaluate(self, weights: Weights) -> Tuple[int, float]:
        return 1, 1.0


class FailingCLient(Client):
    """Test class."""

    def get_weights(self) -> Weights:
        raise Exception()

    def fit(self, weights: Weights) -> Tuple[Weights, int]:
        raise Exception()

    def evaluate(self, weights: Weights) -> Tuple[int, float]:
        raise Exception()


def test_fit_clients() -> None:
    """Test fit_clients."""
    # Prepare
    clients: List[Client] = [
        FailingCLient("0"),
        SuccessClient("1"),
    ]
    weights: Weights = []

    # Execute
    results, failures = fit_clients(clients, weights)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1] == 1


def test_eval_clients() -> None:
    """Test eval_clients."""
    # Prepare
    clients: List[Client] = [
        FailingCLient("0"),
        SuccessClient("1"),
    ]
    weights: Weights = []

    # Execute
    results, failures = eval_clients(clients, weights)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][0] == 1
    assert results[0][1] == 1.0
