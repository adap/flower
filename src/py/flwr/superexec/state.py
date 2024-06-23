from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional


class RunStatus(Enum):
    """RunStatus Enum."""

    RUNNING = auto()
    FINISHED = auto()
    INTERRUPTED = auto()


class SuperexecState(ABC):
    """Abstract SuperexecState."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize state."""

    @abstractmethod
    def store_log(self, run_id: int, log_output: str) -> None:
        """Store a log entry for a given run."""

    @abstractmethod
    def get_logs(self, run_id: int) -> List[str]:
        """Retrieve all log entries for a given run."""

    @abstractmethod
    def update_run_tracker(self, run_id: int, status: RunStatus) -> None:
        """Store or update a RunTracker with the given run_id and status."""

    @abstractmethod
    def get_run_tracker_status(self, run_id: int) -> Optional[RunStatus]:
        """Retrieve the status of a RunTracker by run_id."""
