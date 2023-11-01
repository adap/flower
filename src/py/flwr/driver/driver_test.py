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
from unittest.mock import Mock, patch

from grpc import RpcError, StatusCode

from flwr.driver.driver import Driver
from flwr.proto.driver_pb2 import (
    GetNodesRequest,
    PullTaskResRequest,
    PushTaskInsRequest,
)
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes


def mock_rpc_error(code: StatusCode) -> RpcError:
    """Get an RpcError with the given error code."""
    err = RpcError()
    err.code = lambda: code
    return err


class TestDriver(unittest.TestCase):
    """Tests for `Driver` class."""

    def setUp(self) -> None:
        """Initialize mock GrpcDriver and Driver instance before each test."""
        mock_response = Mock()
        mock_response.workload_id = 61016
        self.mock_grpc_driver = Mock()
        self.mock_grpc_driver.create_workload.return_value = mock_response
        self.patcher = patch(
            "flwr.driver.driver.GrpcDriver", return_value=self.mock_grpc_driver
        )
        self.patcher.start()
        self.driver = Driver()

    def tearDown(self) -> None:
        """Cleanup after each test."""
        self.patcher.stop()

    def test_check_and_init_grpc_driver_already_initialized(self) -> None:
        """Test that GrpcDriver doesn't initialize if workload is created."""
        # Prepare
        self.driver.grpc_driver = self.mock_grpc_driver
        self.driver.workload_id = 61016

        # Execute
        # pylint: disable-next=protected-access
        self.driver._get_grpc_driver_and_workload_id()

        # Assert
        self.mock_grpc_driver.connect.assert_not_called()

    def test_check_and_init_grpc_driver_needs_initialization(self) -> None:
        """Test GrpcDriver initialization when workload is not created."""
        # Execute
        # pylint: disable-next=protected-access
        self.driver._get_grpc_driver_and_workload_id()

        # Assert
        self.mock_grpc_driver.connect.assert_called_once()
        self.assertEqual(self.driver.workload_id, 61016)

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Prepare
        mock_response = Mock()
        mock_response.nodes = [Mock(), Mock()]
        self.mock_grpc_driver.get_nodes.return_value = mock_response

        # Execute
        nodes = self.driver.get_nodes()
        args, kwargs = self.mock_grpc_driver.get_nodes.call_args

        # Assert
        self.mock_grpc_driver.connect.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], GetNodesRequest)
        self.assertEqual(args[0].workload_id, 61016)
        self.assertEqual(nodes, mock_response.nodes)

    def test_push_task_ins(self) -> None:
        """Test pushing task instructions."""
        # Prepare
        mock_response = Mock()
        mock_response.task_ids = ["id1", "id2"]
        self.mock_grpc_driver.push_task_ins.return_value = mock_response
        task_ins_list = [TaskIns(), TaskIns()]

        # Execute
        task_ids = self.driver.push_task_ins(task_ins_list)
        args, kwargs = self.mock_grpc_driver.push_task_ins.call_args

        # Assert
        self.mock_grpc_driver.connect.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PushTaskInsRequest)
        self.assertEqual(task_ids, mock_response.task_ids)
        for task_ins in args[0].task_ins_list:
            self.assertEqual(task_ins.workload_id, 61016)

    def test_pull_task_res_with_given_task_ids(self) -> None:
        """Test pulling task results with specific task IDs."""
        # Prepare
        mock_response = Mock()
        mock_response.task_res_list = [
            TaskRes(task=Task(ancestry=["id2"])),
            TaskRes(task=Task(ancestry=["id3"])),
        ]
        self.mock_grpc_driver.pull_task_res.return_value = mock_response
        task_ids = ["id1", "id2", "id3"]

        # Execute
        task_res_list = self.driver.pull_task_res(task_ids)
        args, kwargs = self.mock_grpc_driver.pull_task_res.call_args

        # Assert
        self.mock_grpc_driver.connect.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PullTaskResRequest)
        self.assertEqual(args[0].task_ids, task_ids)
        self.assertEqual(task_res_list, mock_response.task_res_list)

    def test_del_with_initialized_driver(self) -> None:
        """Test cleanup behavior when Driver is initialized."""
        # Prepare
        # pylint: disable-next=protected-access
        self.driver._get_grpc_driver_and_workload_id()

        # Execute
        self.driver.__del__()

        # Assert
        self.mock_grpc_driver.disconnect.assert_called_once()

    def test_del_with_uninitialized_driver(self) -> None:
        """Test cleanup behavior when Driver is not initialized."""
        # Execute
        self.driver.__del__()

        # Assert
        self.mock_grpc_driver.disconnect.assert_not_called()


class TestDriverRetryMechanism(unittest.TestCase):
    """Test class for verifying the retry mechanism of the Driver."""

    def setUp(self) -> None:
        """Initialize mock GrpcDriver and Driver instance before each test."""
        mock_response = Mock()
        mock_response.workload_id = 61016
        self.mock_grpc_driver = Mock()
        self.mock_grpc_driver.create_workload.return_value = mock_response
        self.patcher = patch(
            "flwr.driver.driver.GrpcDriver", return_value=self.mock_grpc_driver
        )
        self.sleep_patcher = patch("time.sleep", lambda s: None)
        self.patcher.start()
        self.sleep_patcher.start()
        self.driver = Driver()

    def tearDown(self) -> None:
        """Cleanup after each test."""
        self.sleep_patcher.stop()
        self.patcher.stop()

    def test_retry_get_nodes(self) -> None:
        """Test if the `get_nodes` method retries on RpcError."""
        # Prepare
        mock_correct_res = Mock()
        mock_correct_res.nodes = ["a", "b", "c"]
        self.mock_grpc_driver.get_nodes.side_effect = [
            mock_rpc_error(StatusCode.UNAVAILABLE)
        ] * 3 + [mock_correct_res]
        driver = Driver()

        # Execute
        res = driver.get_nodes()

        # Assert
        self.assertEqual(res, mock_correct_res.nodes)
        # `get_nodes` is expected to be called 4 times (3 failures + 1 success)
        self.assertEqual(self.mock_grpc_driver.get_nodes.call_count, 4)

    def test_retry_push_task_ins(self) -> None:
        """Test if the `push_task_ins` method retries on RpcError."""
        # Prepare
        mock_correct_res = Mock()
        mock_correct_res.task_ids = ["a", "b", "c"]
        self.mock_grpc_driver.push_task_ins.side_effect = [
            mock_rpc_error(StatusCode.UNAVAILABLE)
        ] * 3 + [mock_correct_res]
        driver = Driver()

        # Execute
        res = driver.push_task_ins([TaskIns(), TaskIns(), TaskIns()])

        # Assert
        self.assertEqual(res, mock_correct_res.task_ids)
        # `push_task_ins` is expected to be called 4 times (3 failures + 1 success)
        self.assertEqual(self.mock_grpc_driver.push_task_ins.call_count, 4)

    def test_retry_pull_task_res(self) -> None:
        """Test if the `pull_task_res` method retries on RpcError."""
        # Prepare
        mock_correct_res = Mock()
        mock_correct_res.task_res_list = ["resa", "resb", "resc"]
        self.mock_grpc_driver.pull_task_res.side_effect = [
            mock_rpc_error(StatusCode.UNAVAILABLE)
        ] * 3 + [mock_correct_res]
        driver = Driver()

        # Execute
        res = driver.pull_task_res(["a", "b", "c"])

        # Assert
        self.assertEqual(res, mock_correct_res.task_res_list)
        # `pull_task_res` is expected to be called 4 times (3 failures + 1 success)
        self.assertEqual(self.mock_grpc_driver.pull_task_res.call_count, 4)

    def test_retry_get_nodes_and_fail(self) -> None:
        """Test if the `get_nodes` method fail on invalid RpcError."""
        # Prepare
        self.mock_grpc_driver.get_nodes.side_effect = mock_rpc_error(
            StatusCode.UNIMPLEMENTED
        )
        driver = Driver()

        # Execute and assert
        with self.assertRaises(RpcError) as context:
            driver.get_nodes()
        self.mock_grpc_driver.get_nodes.assert_called_once()
        self.assertEqual(context.exception.code(), StatusCode.UNIMPLEMENTED)

    def test_retry_push_task_ins_and_fail(self) -> None:
        """Test if the `push_task_ins` method fail on invalid RpcError."""
        # Prepare
        self.mock_grpc_driver.push_task_ins.side_effect = mock_rpc_error(
            StatusCode.UNIMPLEMENTED
        )
        driver = Driver()

        # Execute and assert
        with self.assertRaises(RpcError) as context:
            driver.push_task_ins([TaskIns()])
        self.mock_grpc_driver.push_task_ins.assert_called_once()
        self.assertEqual(context.exception.code(), StatusCode.UNIMPLEMENTED)

    def test_retry_pull_task_res_and_fail(self) -> None:
        """Test if the `pull_task_res` method fail on invalid RpcError."""
        # Prepare
        self.mock_grpc_driver.pull_task_res.side_effect = mock_rpc_error(
            StatusCode.UNIMPLEMENTED
        )
        driver = Driver()

        # Execute and assert
        with self.assertRaises(RpcError) as context:
            driver.pull_task_res(["a", "b", "c"])
        self.mock_grpc_driver.pull_task_res.assert_called_once()
        self.assertEqual(context.exception.code(), StatusCode.UNIMPLEMENTED)
