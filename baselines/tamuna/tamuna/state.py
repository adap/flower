from abc import ABC, abstractmethod
from typing import Optional, Dict


class ClientState(ABC):
    """Abstract base class for Flower client state."""

    @abstractmethod
    def setup_state(self) -> None:
        """Initialize client state."""

    @abstractmethod
    def fetch_state(
        self,
        timeout: Optional[float],
    ) -> Dict:
        """Return the client's state."""

    @abstractmethod
    def update_state(
        self,
        state: Dict,
        timeout: Optional[float],
    ) -> None:
        """Update the client's state."""
