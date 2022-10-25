# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""Flower driver serialization."""


from typing import List

from flwr.common.serde import (
    client_message_from_proto,
    client_message_to_proto,
    server_message_from_proto,
    server_message_to_proto,
)
from flwr.common.typing import ClientMessage, ServerMessage
from flwr.driver.messages import (
    CreateTasksRequest,
    CreateTasksResponse,
    GetClientsRequest,
    GetClientsResponse,
    GetResultsRequest,
    GetResultsResponse,
    Result,
    Task,
    TaskAssignment,
)
from flwr.proto import driver_pb2, task_pb2

# === GetClients messages ===


def get_clients_request_to_proto(
    req: GetClientsRequest,
) -> driver_pb2.GetClientsRequest:
    """Serialize `GetClientsRequest` to ProtoBuf."""
    # pylint: disable=W0613
    return driver_pb2.GetClientsRequest()


def get_clients_request_from_proto(
    msg: driver_pb2.GetClientsRequest,
) -> GetClientsRequest:
    """Deserialize `GetClientsRequest` from ProtoBuf."""
    # pylint: disable=W0613
    return GetClientsRequest()


def get_clients_response_to_proto(
    res: GetClientsResponse,
) -> driver_pb2.GetClientsResponse:
    """Serialize `GetClientsResponse` to ProtoBuf."""
    return driver_pb2.GetClientsResponse(client_ids=res.client_ids)


def get_clients_response_from_proto(
    msg: driver_pb2.GetClientsResponse,
) -> GetClientsResponse:
    """Deserialize `GetClientsResponse` from ProtoBuf."""
    return GetClientsResponse(client_ids=list(msg.client_ids))


# === CreateTasks messages ===


def create_tasks_request_to_proto(
    req: CreateTasksRequest,
) -> driver_pb2.CreateTasksRequest:
    """Serialize `CreateTasksRequest` to ProtoBuf."""
    task_assignments_proto: List[task_pb2.TaskAssignment] = []
    for task_assignment in req.task_assignments:
        legacy_server_message_proto = (
            None
            if not task_assignment.task.legacy_server_message
            else server_message_to_proto(
                task_assignment.task.legacy_server_message,
            )
        )
        task_proto = task_pb2.Task(
            task_id=task_assignment.task.task_id,
            legacy_server_message=legacy_server_message_proto,
        )
        task_assignment_proto = task_pb2.TaskAssignment(
            task=task_proto,
            client_ids=task_assignment.client_ids,
        )
        task_assignments_proto.append(task_assignment_proto)
    return driver_pb2.CreateTasksRequest(task_assignments=task_assignments_proto)


def create_tasks_request_from_proto(
    msg: driver_pb2.CreateTasksRequest,
) -> CreateTasksRequest:
    """Deserialize `CreateTasksRequest` from ProtoBuf."""
    task_assignments: List[TaskAssignment] = []
    for task_assignment_proto in msg.task_assignments:
        legacy_server_message = (
            ServerMessage()  # Empty ServerMessage
            if not task_assignment_proto.task.legacy_server_message
            else server_message_from_proto(
                task_assignment_proto.task.legacy_server_message,
            )
        )
        task = Task(
            task_id=task_assignment_proto.task.task_id,
            legacy_server_message=legacy_server_message,
        )
        task_assignment = TaskAssignment(
            task=task,
            client_ids=list(task_assignment_proto.client_ids),
        )
        task_assignments.append(task_assignment)
    return CreateTasksRequest(task_assignments=task_assignments)


def create_tasks_response_to_proto(
    res: CreateTasksResponse,
) -> driver_pb2.CreateTasksResponse:
    """Serialize `CreateTasksResponse` to ProtoBuf."""
    return driver_pb2.CreateTasksResponse(task_ids=res.task_ids)


def create_tasks_response_from_proto(
    msg: driver_pb2.CreateTasksResponse,
) -> CreateTasksResponse:
    """Deserialize `CreateTasksResponse` from ProtoBuf."""
    return CreateTasksResponse(task_ids=list(msg.task_ids))


# === GetResults messages ===


def get_results_request_to_proto(
    req: GetResultsRequest,
) -> driver_pb2.GetResultsRequest:
    """Serialize `GetResultsRequest` to ProtoBuf."""
    return driver_pb2.GetResultsRequest(task_ids=req.task_ids)


def get_results_request_from_proto(
    msg: driver_pb2.GetResultsRequest,
) -> GetResultsRequest:
    """Deserialize `GetResultsRequest` from ProtoBuf."""
    return GetResultsRequest(task_ids=list(msg.task_ids))


def get_results_response_to_proto(
    res: GetResultsResponse,
) -> driver_pb2.GetResultsResponse:
    """Serialize `GetResultsResponse` to ProtoBuf."""
    results_proto: List[task_pb2.Result] = []
    for result in res.results:
        legacy_client_message_proto = (
            None
            if not result.legacy_client_message
            else client_message_to_proto(result.legacy_client_message)
        )
        result_proto = task_pb2.Result(
            task_id=result.task_id,
            legacy_client_message=legacy_client_message_proto,
        )
        results_proto.append(result_proto)
    return driver_pb2.GetResultsResponse(results=results_proto)


def get_results_response_from_proto(
    msg: driver_pb2.GetResultsResponse,
) -> GetResultsResponse:
    """Deserialize `GetResultsResponse` from ProtoBuf."""
    results: List[Result] = []
    for result_proto in msg.results:
        legacy_client_message = (
            ClientMessage()  # Empty ServerMessage
            if not result_proto.legacy_client_message
            else client_message_from_proto(result_proto.legacy_client_message)
        )
        result = Result(
            task_id=result_proto.task_id,
            legacy_client_message=legacy_client_message,
        )
        results.append(result)
    return GetResultsResponse(results=results)
