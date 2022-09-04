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
from abc import ABC, abstractmethod
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,

)
from flwr.common.typing import SAServerMessageCarrier, SAClientMessageCarrier
from flwr.client.client import Client


class SAClientWrapper(Client, ABC):
    """Wrapper which adds SecAgg methods."""

    def __init__(self, c: Client) -> None:
        self.client = c
        self.__sec_id = None

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        return self.client.get_parameters()

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    @abstractmethod
    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        """response to the server for Secure Aggregation"""

    def get_sec_id(self) -> int:
        if self.__sec_id:
            return self.__sec_id
        raise Exception("Unassigned secure ID")

    def set_sec_id(self, idx: int):
        self.__sec_id = idx
