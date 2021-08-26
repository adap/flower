# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
from flwr.common.sec_agg import sec_agg_client_logic
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

    def setup_param(self, setup_param_ins: SetupParamIns):
        return sec_agg_client_logic.setup_param(self, setup_param_ins)

    def ask_keys(self, ask_keys_ins: AskKeysIns) -> AskKeysRes:
        return sec_agg_client_logic.ask_keys(self, ask_keys_ins)

    def share_keys(self, share_keys_in: ShareKeysIns) -> ShareKeysRes:
        return sec_agg_client_logic.share_keys(self, share_keys_in)

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        return sec_agg_client_logic.ask_vectors(self, ask_vectors_ins)

    def unmask_vectors(self, unmask_vectors_ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
        return sec_agg_client_logic.unmask_vectors(self, unmask_vectors_ins)
