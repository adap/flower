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
"""Flower server strategy."""


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from flower.typing import Weights


class Strategy(ABC):
    """Abstract class to implement custom server strategies."""

    def __init__(self) -> None:
        self.current_round: int = 0

    def next_round(self) -> None:
        """Inform the strategy implementation that the next round of FL has begun."""
        self.current_round += 1

    @abstractmethod
    def should_evaluate(self) -> bool:
        """Decide if the current global model should be evaluated or not."""

    @abstractmethod
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine the number of clients used for training."""

    @abstractmethod
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine the number of clients used for evaluation."""

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate the current model weights."""

    @abstractmethod
    def on_aggregate_fit(
        self, results: List[Tuple[Weights, int]], failures: List[BaseException]
    ) -> Optional[Weights]:
        """Aggregate training results."""

    @abstractmethod
    def on_aggregate_evaluate(
        self, results: List[Tuple[int, float]], failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation results."""
