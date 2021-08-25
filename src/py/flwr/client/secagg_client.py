from typing import Dict, List, Tuple
import numpy as np
from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,

)
from flwr.common.parameter import parameters_to_weights, weights_to_parameters
from flwr.common.typing import AskKeysIns, AskVectorsIns, AskVectorsRes, SetupParamIns, SetupParamRes, ShareKeysIns, ShareKeysPacket, ShareKeysRes, UnmaskVectorsIns, UnmaskVectorsRes, Weights
from flwr.common.secagg import secagg_primitives, secagg_client_logic
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

    def setup_param(self, setup_param_ins: SetupParamIns):
        return secagg_client_logic.setup_param(self, setup_param_ins)

    def ask_keys(self, ask_keys_ins: AskKeysIns) -> AskKeysRes:
        return secagg_client_logic.ask_keys(self, ask_keys_ins)

    def share_keys(self, share_keys_in: ShareKeysIns) -> ShareKeysRes:
        return secagg_client_logic.share_keys(self, share_keys_in)

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        return secagg_client_logic.ask_vectors(self, ask_vectors_ins)

    def unmask_vectors(self, unmask_vectors_ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
        return secagg_client_logic.unmask_vectors(self, unmask_vectors_ins)
