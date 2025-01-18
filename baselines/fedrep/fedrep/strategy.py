"""fedrep: A Flower Baseline."""

from flwr.server.strategy import FedAvg


class FedRep(FedAvg):
    """FedRep strategy."""

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedRep(accept_failures={self.accept_failures})"
        return rep
