"""Factory class that creates NodeState instances."""

from typing import Optional

from .in_memory_nodestate import InMemoryNodeState
from .nodestate import NodeState


class NodeStateFactory:
    """Factory class that creates NodeState instances."""

    def __init__(self) -> None:
        self.state_instance: Optional[NodeState] = None

    def state(self) -> NodeState:
        """Return a State instance and create it, if necessary."""
        if self.state_instance is None:
            self.state_instance = InMemoryNodeState()
        return self.state_instance
