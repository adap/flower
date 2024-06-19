# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Status,
)
from flwr.server.client_proxy import ClientProxy


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetPropertiesRes:
        """Return the client's properties."""
        return GetPropertiesRes(status=Status(code=Code.OK, message=""), properties={})

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetParametersRes:
        """Return the current local model parameters."""
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        return FitRes(
            status=Status(Code.OK, message=""),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=0,
            metrics={},
        )

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        return EvaluateRes(
            status=Status(Code.OK, message=""), loss=0.0, num_examples=0, metrics={}
        )

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return DisconnectRes(reason="")


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