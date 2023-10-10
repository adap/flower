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
"""Tests for driver SDK."""

import unittest
from unittest.mock import MagicMock

from flwr.driver.driver import Driver
from flwr.proto.driver_pb2 import (
    CreateWorkloadRequest,
    CreateWorkloadResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskIns, TaskRes


# mypy: disable-error-code=union-attr
class TestDriverMethods(unittest.TestCase):
    """Tests for Driver."""

    def setUp(self) -> None:
        """Set up."""
        self.driver = Driver()
        # Setting stub and channel to MagicMock objects
        self.driver.stub = MagicMock()
        self.driver.channel = MagicMock()

    def test_create_workload(self) -> None:
        """Test `create_workload()`.

        Test if the method correctly sets `Driver.workload_id` and returns
        the response.
        """
        # Prepare
        self.driver.stub.CreateWorkload.return_value = CreateWorkloadResponse(
            workload_id=123
        )

        # Execute
        response = self.driver.create_workload(CreateWorkloadRequest())

        # Assert
        self.assertEqual(response.workload_id, 123)
        self.assertEqual(self.driver.workload_id, 123)

    def test_get_nodes(self) -> None:
        """Test `get_nodes()`.

        Test if the method correctly sets the `workload_id` field and returns the
        response.
        """
        # Prepare
        self.driver.workload_id = 123
        request = GetNodesRequest()
        nodes = [
            Node(node_id=-321, anonymous=False),
            Node(node_id=123, anonymous=False),
        ]
        mock_response = GetNodesResponse(nodes=nodes)

        self.driver.stub.GetNodes.return_value = mock_response
        # Execute
        response = self.driver.get_nodes(request)

        # Assert
        # pylint: disable-next=no-member
        self.assertEqual(request.workload_id, 123)
        self.assertEqual(list(response.nodes), nodes)

    def test_push_task_ins(self) -> None:
        """Test `push_task_ins()`.

        Test if the method correctly sets `workload_id` fields in TaskIns and
        returns the response.
        """
        # Prepare
        self.driver.workload_id = 123
        request = PushTaskInsRequest(task_ins_list=[TaskIns()])
        mock_response = PushTaskInsResponse(task_ids=["task id 123"])
        self.driver.stub.PushTaskIns.return_value = mock_response

        # Execute
        response = self.driver.push_task_ins(request)

        # Assert
        # pylint: disable-next=no-member
        self.assertEqual(response.task_ids, ["task id 123"])
        # pylint: disable-next=no-member
        self.assertEqual(request.task_ins_list[0].workload_id, 123)

    def test_pull_task_res(self) -> None:
        """Test if `pull_task_res()` correctly returns the response."""
        # Prepare
        self.driver.workload_id = 123
        task_res = TaskRes(task_id="task id 123")
        mock_response = PullTaskResResponse(task_res_list=[task_res])
        self.driver.stub.PullTaskRes.return_value = mock_response

        # Execute
        response = self.driver.pull_task_res(
            PullTaskResRequest(task_ids=["task id 123"])
        )

        # Assert
        self.assertEqual(list(response.task_res_list), [task_res])

    def test_create_workload_error_not_initialized(self) -> None:
        """Test `create_workload()` failure due to Driver not connected."""
        # Prepare
        self.driver.stub = None

        # Assert
        with self.assertRaises(Exception) as context:
            self.driver.create_workload(CreateWorkloadRequest())

        self.assertIn("`Driver` instance not connected", str(context.exception))

    def test_method_error_no_workload(self) -> None:
        """Test method failure due to workload not created."""
        # Prepare
        self.driver.workload_id = None

        # Assert
        with self.assertRaises(Exception) as context:
            self.driver.get_nodes(GetNodesRequest())

        self.assertIn("`Driver` instance not initialized", str(context.exception))
