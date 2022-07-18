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
"""Tests for Flower ClientProxy."""


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
    ReconnectIns,
)
from flwr.server.client_proxy import ClientProxy


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        """Returns the client's properties."""

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
    ) -> GetParametersRes:
        """Return the current local model parameters."""

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
    ) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
    ) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""


def test_cid() -> None:
    """Tests if the register method works correctly."""
    # Prepare
    cid_expected = "1"
    client_proxy = CustomClientProxy(cid=cid_expected)

    # Execute
    cid_actual = client_proxy.cid

    # Assert
    assert cid_actual == cid_expected


def test_properties_are_empty() -> None:
    """Tests if the register method works correctly."""
    # Prepare
    client_proxy = CustomClientProxy(cid="1")

    # Execute
    actual_properties = client_proxy.properties

    # Assert
    assert not actual_properties
