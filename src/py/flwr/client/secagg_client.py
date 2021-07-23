from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    secagg_utils,
)
from .client import Client


class SecAggClient(Client):
    """Wrapper which adds SecAgg methods."""

    def __init__(self, c: Client) -> None:
        self.client = c

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        return self.client.get_parameters()

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    def ask_keys(self):
        self.sk1, self.pk1 = secagg_utils.generate_key_pairs()
        self.sk2, self.pk2 = secagg_utils.generate_key_pairs()
        return AskKeysRes(
            pk1=secagg_utils.public_key_to_bytes(self.pk1),
            pk2=secagg_utils.public_key_to_bytes(self.pk2),
        )
