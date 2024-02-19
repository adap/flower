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
"""Client-side message handler tests."""


import uuid

from flwr.client import Client
from flwr.client.typing import ClientFn
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
    Parameters,
    Status,
)
from flwr.common import recordset_compat as compat
from flwr.common import typing
from flwr.common.constant import MESSAGE_TYPE_GET_PROPERTIES
from flwr.common.context import Context
from flwr.common.message import Message, Metadata
from flwr.common.recordset import RecordSet

from .message_handler import handle_legacy_message_from_msgtype


class ClientWithoutProps(Client):
    """Client not implementing get_properties."""

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


class ClientWithProps(Client):
    """Client implementing get_properties."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Get fixed properties of the client with 'Success' status."""
        return GetPropertiesRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            properties={"str_prop": "val", "int_prop": 1},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


def _get_client_fn(client: Client) -> ClientFn:
    def client_fn(cid: str) -> Client:  # pylint: disable=unused-argument
        return client

    return client_fn


def test_client_without_get_properties() -> None:
    """Test client implementing get_properties."""
    # Prepare
    client = ClientWithoutProps()
    recordset = compat.getpropertiesins_to_recordset(GetPropertiesIns({}))
    message = Message(
        metadata=Metadata(
            run_id=0,
            message_id=str(uuid.uuid4()),
            group_id="",
            node_id=0,
            ttl="",
            message_type=MESSAGE_TYPE_GET_PROPERTIES,
        ),
        content=recordset,
    )

    # Execute
    actual_msg = handle_legacy_message_from_msgtype(
        client_fn=_get_client_fn(client),
        message=message,
        context=Context(state=RecordSet()),
    )

    # Assert
    expected_get_properties_res = GetPropertiesRes(
        status=Status(
            code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message="Client does not implement `get_properties`",
        ),
        properties={},
    )
    expected_rs = compat.getpropertiesres_to_recordset(expected_get_properties_res)
    expected_msg = Message(message.metadata, expected_rs)

    assert actual_msg.content == expected_msg.content
    assert actual_msg.metadata.message_type == expected_msg.metadata.message_type


def test_client_with_get_properties() -> None:
    """Test client not implementing get_properties."""
    # Prepare
    client = ClientWithProps()
    recordset = compat.getpropertiesins_to_recordset(GetPropertiesIns({}))
    message = Message(
        metadata=Metadata(
            run_id=0,
            message_id=str(uuid.uuid4()),
            group_id="",
            node_id=0,
            ttl="",
            message_type=MESSAGE_TYPE_GET_PROPERTIES,
        ),
        content=recordset,
    )

    # Execute
    actual_msg = handle_legacy_message_from_msgtype(
        client_fn=_get_client_fn(client),
        message=message,
        context=Context(state=RecordSet()),
    )

    # Assert
    expected_get_properties_res = GetPropertiesRes(
        status=Status(
            code=Code.OK,
            message="Success",
        ),
        properties={"str_prop": "val", "int_prop": 1},
    )
    expected_rs = compat.getpropertiesres_to_recordset(expected_get_properties_res)
    expected_msg = Message(message.metadata, expected_rs)

    assert actual_msg.content == expected_msg.content
    assert actual_msg.metadata.message_type == expected_msg.metadata.message_type
