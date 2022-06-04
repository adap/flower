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
"""Flower Client tests."""


from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
)

from .client import Client, has_get_properties


class OverridingClient(Client):
    """Client overriding `get_properties`."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"), properties={}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # This method is not expected to be called
        raise Exception()

    def fit(self, ins: FitIns) -> FitRes:
        # This method is not expected to be called
        raise Exception()

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # This method is not expected to be called
        raise Exception()


class NotOverridingClient(Client):
    """Client not overriding `get_properties`."""

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # This method is not expected to be called
        raise Exception()

    def fit(self, ins: FitIns) -> FitRes:
        # This method is not expected to be called
        raise Exception()

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # This method is not expected to be called
        raise Exception()


def test_has_get_properties_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_properties_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected
