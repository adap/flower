# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod
from typing import Optional

from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
    ReconnectIns,
)


class ClientProxy(ABC):
    """Abstract base class for Flower client proxies."""

    node_id: int

    def __init__(self, cid: str):
        self.cid = cid
        self.properties: Properties = {}

    @abstractmethod
    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        """Return the client's properties."""

    @abstractmethod
    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
    ) -> GetParametersRes:
        """Return the current local model parameters."""

    @abstractmethod
    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
    ) -> FitRes:
        """Refine the provided parameters using the locally held dataset."""

    @abstractmethod
    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
    ) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""

    @abstractmethod
    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
