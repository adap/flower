# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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


import time
import unittest
from unittest.mock import Mock, patch

import grpc

from flwr.common import DEFAULT_TTL, RecordSet
from flwr.common.message import Error
from flwr.common.serde import error_to_proto, recordset_to_proto
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    PullTaskResRequest,
    PushTaskInsRequest,
)
from flwr.proto.task_pb2 import Task, TaskRes  # pylint: disable=E0611

from .grpc_driver import GrpcDriver


class TestGrpcDriver(unittest.TestCase):
    """Tests for `GrpcDriver` class."""

    def setUp(self) -> None:
        """Initialize mock GrpcDriverStub and Driver instance before each test."""

        def _mock_fn(req: GetRunRequest) -> GetRunResponse:
            return GetRunResponse(
                run=Run(
                    run_id=req.run_id,
                    fab_id="mock/mock",
                    fab_version="v1.0.0",
                    fab_hash="9f86d08",
                )
            )

        self.mock_stub = Mock()
        self.mock_channel = Mock()
        self.mock_stub.GetRun.side_effect = _mock_fn
        self.driver = GrpcDriver()
        self.driver._grpc_stub = self.mock_stub  # pylint: disable=protected-access
        self.driver._channel = self.mock_channel  # pylint: disable=protected-access
        self.driver.set_run(run_id=61016)

    def test_init_grpc_driver(self) -> None:
        """Test GrpcDriverStub initialization."""
        # Assert
        self.assertEqual(self.driver.run.run_id, 61016)
        self.assertEqual(self.driver.run.fab_id, "mock/mock")
        self.assertEqual(self.driver.run.fab_version, "v1.0.0")
        self.assertEqual(self.driver.run.fab_hash, "9f86d08")
        self.mock_stub.GetRun.assert_called_once()

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Prepare
        mock_response = Mock()
        mock_response.nodes = [Mock(node_id=404), Mock(node_id=200)]
        self.mock_stub.GetNodes.return_value = mock_response

        # Execute
        node_ids = self.driver.get_node_ids()
        args, kwargs = self.mock_stub.GetNodes.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], GetNodesRequest)
        self.assertEqual(args[0].run_id, 61016)
        self.assertEqual(node_ids, [404, 200])

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        mock_response = Mock(task_ids=["id1", "id2"])
        self.mock_stub.PushTaskIns.return_value = mock_response
        msgs = [
            self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)
            for _ in range(2)
        ]

        # Execute
        msg_ids = self.driver.push_messages(msgs)
        args, kwargs = self.mock_stub.PushTaskIns.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PushTaskInsRequest)
        self.assertEqual(msg_ids, mock_response.task_ids)
        for task_ins in args[0].task_ins_list:
            self.assertEqual(task_ins.run_id, 61016)

    def test_push_messages_invalid(self) -> None:
        """Test pushing invalid messages."""
        # Prepare
        mock_response = Mock(task_ids=["id1", "id2"])
        self.mock_stub.PushTaskIns.return_value = mock_response
        msgs = [
            self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)
            for _ in range(2)
        ]
        # Use invalid run_id
        msgs[1].metadata.__dict__["_run_id"] += 1  # pylint: disable=protected-access

        # Execute and assert
        with self.assertRaises(ValueError):
            self.driver.push_messages(msgs)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        mock_response = Mock()
        # A Message must have either content or error set so we prepare
        # two tasks that contain these.
        mock_response.task_res_list = [
            TaskRes(
                task=Task(ancestry=["id2"], recordset=recordset_to_proto(RecordSet()))
            ),
            TaskRes(task=Task(ancestry=["id3"], error=error_to_proto(Error(code=0)))),
        ]
        self.mock_stub.PullTaskRes.return_value = mock_response
        msg_ids = ["id1", "id2", "id3"]

        # Execute
        msgs = self.driver.pull_messages(msg_ids)
        reply_tos = {msg.metadata.reply_to_message for msg in msgs}
        args, kwargs = self.mock_stub.PullTaskRes.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PullTaskResRequest)
        self.assertEqual(args[0].task_ids, msg_ids)
        self.assertEqual(reply_tos, {"id2", "id3"})

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        mock_response = Mock(task_ids=["id1"])
        self.mock_stub.PushTaskIns.return_value = mock_response
        # The response message must include either `content` (i.e. a recordset) or
        # an `Error`. We choose the latter in this case
        error_proto = error_to_proto(Error(code=0))
        mock_response = Mock(
            task_res_list=[TaskRes(task=Task(ancestry=["id1"], error=error_proto))]
        )
        self.mock_stub.PullTaskRes.return_value = mock_response
        msgs = [self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)]

        # Execute
        ret_msgs = list(self.driver.send_and_receive(msgs))

        # Assert
        self.assertEqual(len(ret_msgs), 1)
        self.assertEqual(ret_msgs[0].metadata.reply_to_message, "id1")

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        sleep_fn = time.sleep
        mock_response = Mock(task_ids=["id1"])
        self.mock_stub.PushTaskIns.return_value = mock_response
        mock_response = Mock(task_res_list=[])
        self.mock_stub.PullTaskRes.return_value = mock_response
        msgs = [self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)]

        # Execute
        with patch("time.sleep", side_effect=lambda t: sleep_fn(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.driver.send_and_receive(msgs, timeout=0.15))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_del_with_initialized_driver(self) -> None:
        """Test cleanup behavior when Driver is initialized."""
        # Execute
        self.driver.close()

        # Assert
        self.mock_channel.close.assert_called_once()

    def test_del_with_uninitialized_driver(self) -> None:
        """Test cleanup behavior when Driver is not initialized."""
        # Prepare
        self.driver._grpc_stub = None  # pylint: disable=protected-access
        self.driver._channel = None  # pylint: disable=protected-access

        # Execute
        self.driver.close()

        # Assert
        self.mock_channel.close.assert_not_called()

    def test_simple_retry_mechanism_get_nodes(self) -> None:
        """Test retry mechanism with the get_node_ids method."""
        # Prepare
        grpc_exc = grpc.RpcError()
        grpc_exc.code = lambda: grpc.StatusCode.UNAVAILABLE
        mock_get_nodes = Mock()
        mock_get_nodes.side_effect = [
            grpc_exc,
            Mock(nodes=[Mock(node_id=404)]),
        ]
        # Make pylint happy
        # pylint: disable=protected-access
        self.driver._grpc_stub = Mock(
            GetNodes=lambda *args, **kwargs: self.driver._retry_invoker.invoke(
                mock_get_nodes, *args, **kwargs
            )
        )
        # pylint: enable=protected-access

        # Execute
        with patch("time.sleep", side_effect=lambda _: None):
            node_ids = self.driver.get_node_ids()

        # Assert
        self.assertIn(404, node_ids)
        self.assertEqual(mock_get_nodes.call_count, 2)
