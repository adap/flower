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
from typing import Dict, List, Optional, Tuple

from flower.client_manager import ClientManager
from flower.client_proxy import ClientProxy
from flower.typing import EvaluateRes, FitIns, FitRes, Weights


class Strategy(ABC):
    """Abstract class to implement custom server strategies."""

    @abstractmethod
    def should_evaluate(self) -> bool:
        """Decide if the current global model should be evaluated or not."""

    @abstractmethod
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine the number of clients used for evaluation."""

    @abstractmethod
    def on_evaluate_config(self, rnd: int) -> Dict[str, str]:
        """Get configuration for the next round of evaluation."""

    @abstractmethod
    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

    @abstractmethod
    def on_aggregate_fit(
        self, results: List[FitRes], failures: List[BaseException]
    ) -> Optional[Weights]:
        """Aggregate training results."""

    @abstractmethod
    def on_aggregate_evaluate(
        self, results: List[EvaluateRes], failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation results."""

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate the current model weights."""

    @abstractmethod
    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Conclude federated learning round and decide whether to continue or not."""
