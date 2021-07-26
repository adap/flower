from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    secagg_utils,
)
from flwr.common.typing import SetupParamIn
from flwr.server.strategy import secagg
from .client import Client
from flwr.common.logger import log
from logging import DEBUG, INFO, WARNING


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

    def setup_param(self, setup_param_in: SetupParamIn):
        self.param = setup_param_in
        log(INFO, f"SecAgg Params: {self.param}")

    def ask_keys(self):
        self.sk1, self.pk1 = secagg_utils.generate_key_pairs()
        self.sk2, self.pk2 = secagg_utils.generate_key_pairs()
        log(INFO, "Created SecAgg Key Pairs")
        return AskKeysRes(
            pk1=secagg_utils.public_key_to_bytes(self.pk1),
            pk2=secagg_utils.public_key_to_bytes(self.pk2),
        )
