
from abc import ABC, abstractmethod
from typing import Optional
from flwr.server import Grid
from flwr.common import Message, ArrayRecord, RecordDict

class Strategy(ABC):
    """Abstract base class for server strategy implementations."""

    @abstractmethod
    def initialize_parameters(
        self, grid: Grid
    ) -> Optional[ArrayRecord]:
        """Initialize the (global) model parameters."""

    @abstractmethod
    def configure_train(
        self, server_round: int, record: RecordDict, grid: Grid
    ) -> list[Message]:
        """Configure the next round of training."""

    @abstractmethod
    def aggregate_train(
        self,
        server_round: int,
        results: list[Message],
        failures: list[Message],
    ) -> RecordDict:
        """Aggregate training results."""

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, record: RecordDict, grid: Grid
    ) -> list[Message]:
        """Configure the next round of evaluation."""

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[Message],
        failures: list[Message],
    ) -> RecordDict:
        """Aggregate evaluation results."""

    @abstractmethod
    def evaluate(
        self, server_round: int, record: RecordDict
    ) -> RecordDict:
        """Evaluate the current model parameters."""