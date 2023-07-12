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


import uuid

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
    Parameters,
    serde,
    typing,
)
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, Code, ServerMessage, Status

from .message_handler import handle


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


def test_client_without_get_properties() -> None:
    """Test client implementing get_properties."""
    # Prepare
    client = ClientWithoutProps()
    ins = ServerMessage.GetPropertiesIns()

    task_ins: TaskIns = TaskIns(
        task_id=str(uuid.uuid4()),
        group_id="",
        workload_id="",
        task=Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=[],
            legacy_server_message=ServerMessage(get_properties_ins=ins),
        ),
    )

    # Execute
    task_res, actual_sleep_duration, actual_keep_going = handle(
        client=client, task_ins=task_ins
    )

    if not task_res.HasField("task"):
        raise ValueError("Task value not found")

    # pylint: disable=no-member
    if not task_res.task.HasField("legacy_client_message"):
        raise ValueError("Unexpected None value")
    # pylint: enable=no-member

    task_res.MergeFrom(
        TaskRes(
            task_id=str(uuid.uuid4()),
            group_id="",
            workload_id="",
        )
    )
    # pylint: disable=no-member
    task_res.task.MergeFrom(
        Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=[task_ins.task_id],
        )
    )

    actual_msg = task_res.task.legacy_client_message
    # pylint: enable=no-member

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
    task_ins = TaskIns(
        task_id=str(uuid.uuid4()),
        group_id="",
        workload_id="",
        task=Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=[],
            legacy_server_message=ServerMessage(get_properties_ins=ins),
        ),
    )

    # Execute
    task_res, actual_sleep_duration, actual_keep_going = handle(
        client=client, task_ins=task_ins
    )

    if not task_res.HasField("task"):
        raise ValueError("Task value not found")

    # pylint: disable=no-member
    if not task_res.task.HasField("legacy_client_message"):
        raise ValueError("Unexpected None value")
    # pylint: enable=no-member

    task_res.MergeFrom(
        TaskRes(
            task_id=str(uuid.uuid4()),
            group_id="",
            workload_id="",
        )
    )
    # pylint: disable=no-member
    task_res.task.MergeFrom(
        Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=[task_ins.task_id],
        )
    )

    actual_msg = task_res.task.legacy_client_message
    # pylint: enable=no-member

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
