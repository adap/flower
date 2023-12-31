"""FedPara strategy."""


from flwr.server.strategy import FedAvg


class FedPara(FedAvg):
    """FedPara strategy."""

    def __init__(
        self,
        algorithm: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def __repr__(self) -> str:
        """Return the name of the strategy."""
        return self.algorithm
