from logging import DEBUG
from typing import Optional

from flwr.common import log

from .state import SuperexecState
from .sqlite_state import SqliteSuperexecState
from .in_memory_state import InMemorySuperexecState


class SuperexecStateFactory:
    """Factory class that creates State instances."""

    def __init__(self, database: str) -> None:
        self.database = database
        self.state_instance: Optional[SuperexecState] = None

    def state(self) -> SuperexecState:
        """Return a State instance and create it, if necessary."""
        # InMemoryState
        if self.database == ":flwr-in-memory-state:":
            if self.state_instance is None:
                self.state_instance = InMemorySuperexecState()
            log(DEBUG, "Using InMemoryState")
            return self.state_instance

        # SqliteState
        state = SqliteSuperexecState(self.database)
        state.initialize()
        log(DEBUG, "Using SqliteState")
        return state
