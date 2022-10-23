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
"""Client-side message handler tests."""


from flwr.client import Client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    serde,
    typing,
)
from flwr.proto.transport_pb2 import ClientMessage, Code, ServerMessage, Status

from .message_handler import handle


class ClientWithoutProps(Client):
    """Client not implementing get_properties."""

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        pass

    def fit(self, ins: FitIns) -> FitRes:
        pass

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        pass


class ClientWithProps(Client):
    """Client implementing get_properties."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            properties={"str_prop": "val", "int_prop": 1},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        pass

    def fit(self, ins: FitIns) -> FitRes:
        pass

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        pass


def test_client_without_get_properties() -> None:
    """Test client implementing get_properties."""
    # Prepare
    client = ClientWithoutProps()
    ins = ServerMessage.GetPropertiesIns()
    msg = ServerMessage(get_properties_ins=ins)

    # Execute
    actual_msg, actual_sleep_duration, actual_keep_going = handle(
        client=client, server_msg=msg
    )

    # Assert
    expected_get_properties_res = ClientMessage.GetPropertiesRes(
        status=Status(
            code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message="Client does not implement `get_properties`",
        )
    )
    expected_msg = ClientMessage(get_properties_res=expected_get_properties_res)

    assert actual_msg == expected_msg
    assert actual_sleep_duration == 0
    assert actual_keep_going is True


def test_client_with_get_properties() -> None:
    """Test client not implementing get_properties."""
    # Prepare
    client = ClientWithProps()
    ins = ServerMessage.GetPropertiesIns()
    msg = ServerMessage(get_properties_ins=ins)

    # Execute
    actual_msg, actual_sleep_duration, actual_keep_going = handle(
        client=client, server_msg=msg
    )

    # Assert
    expected_get_properties_res = ClientMessage.GetPropertiesRes(
        status=Status(
            code=Code.OK,
            message="Success",
        ),
        properties=serde.properties_to_proto(
            properties={"str_prop": "val", "int_prop": 1}
        ),
    )
    expected_msg = ClientMessage(get_properties_res=expected_get_properties_res)

    assert actual_msg == expected_msg
    assert actual_sleep_duration == 0
    assert actual_keep_going is True
