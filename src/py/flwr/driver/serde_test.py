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
"""Tests for driver serialization."""


from flwr.driver import serde
from flwr.driver.messages import (
    CreateTasksRequest,
    CreateTasksResponse,
    GetClientsRequest,
    GetClientsResponse,
    GetResultsRequest,
    GetResultsResponse,
)


def test_get_clients_request_serde() -> None:
    """Test `GetClientsRequest` (de-)serialization."""

    # Prepare
    req = GetClientsRequest()

    # Execute
    msg = serde.get_clients_request_to_proto(req)
    req_actual = serde.get_clients_request_from_proto(msg)

    # Assert
    assert req_actual == req


def test_get_clients_response_serde() -> None:
    """Test `GetClientsResponse` (de-)serialization."""

    # Prepare
    res = GetClientsResponse(client_ids=[1, 2, 3])

    # Execute
    msg = serde.get_clients_response_to_proto(res)
    res_actual = serde.get_clients_response_from_proto(msg)

    # Assert
    assert res_actual == res


def test_create_tasks_request_serde() -> None:
    """Test `CreateTasksRequest` (de-)serialization."""

    # Prepare
    res = CreateTasksRequest(task_assignments=[])

    # Execute
    msg = serde.create_tasks_request_to_proto(res)
    res_actual = serde.create_tasks_request_from_proto(msg)

    # Assert
    assert res_actual == res


def test_create_tasks_response_serde() -> None:
    """Test `CreateTasksResponse` (de-)serialization."""

    # Prepare
    res = CreateTasksResponse(task_ids=[1, 2, 3])

    # Execute
    msg = serde.create_tasks_response_to_proto(res)
    res_actual = serde.create_tasks_response_from_proto(msg)

    # Assert
    assert res_actual == res


def test_get_results_request_serde() -> None:
    """Test `GetResultsRequest` (de-)serialization."""

    # Prepare
    res = GetResultsRequest(task_ids=[])

    # Execute
    msg = serde.get_results_request_to_proto(res)
    res_actual = serde.get_results_request_from_proto(msg)

    # Assert
    assert res_actual == res


def test_get_results_response_serde() -> None:
    """Test `GetResultsResponse` (de-)serialization."""

    # Prepare
    res = GetResultsResponse(results=[])

    # Execute
    msg = serde.get_results_response_to_proto(res)
    res_actual = serde.get_results_response_from_proto(msg)

    # Assert
    assert res_actual == res
