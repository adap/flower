# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for grid SDK."""


import time
import unittest
from unittest.mock import Mock, patch

import grpc

from flwr.common import RecordDict
from flwr.common.message import Error, Message
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    PullResMessagesRequest,
    PushInsMessagesRequest,
)

from ..superlink.linkstate.linkstate_test import create_res_message
from .grpc_grid import GrpcGrid


class TestGrpcGrid(unittest.TestCase):
    """Tests for `GrpcGrid` class."""

    def setUp(self) -> None:
        """Initialize mock GrpcServerAppIoStub and Grid instance before each test."""

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
        self.grid = GrpcGrid()
        self.grid._grpc_stub = self.mock_stub  # pylint: disable=protected-access
        self.grid._channel = self.mock_channel  # pylint: disable=protected-access
        self.grid.set_run(run_id=61016)

    def test_init_grpc_grid(self) -> None:
        """Test GrpcServerAppIoStub initialization."""
        # Assert
        self.assertEqual(self.grid.run.run_id, 61016)
        self.assertEqual(self.grid.run.fab_id, "mock/mock")
        self.assertEqual(self.grid.run.fab_version, "v1.0.0")
        self.assertEqual(self.grid.run.fab_hash, "9f86d08")
        self.mock_stub.GetRun.assert_called_once()

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Prepare
        mock_response = Mock()
        mock_response.nodes = [Mock(node_id=404), Mock(node_id=200)]
        self.mock_stub.GetNodes.return_value = mock_response

        # Execute
        node_ids = self.grid.get_node_ids()
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
        mock_response = Mock(message_ids=["id1", "id2"])
        self.mock_stub.PushMessages.return_value = mock_response
        msgs = [Message(RecordDict(), 0, "query") for _ in range(2)]

        # Execute
        msg_ids = self.grid.push_messages(msgs)
        args, kwargs = self.mock_stub.PushMessages.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PushInsMessagesRequest)
        self.assertEqual(msg_ids, mock_response.message_ids)
        for message in args[0].messages_list:
            self.assertEqual(message.metadata.run_id, 61016)

    def test_push_messages_invalid(self) -> None:
        """Test pushing invalid messages."""
        # Prepare
        mock_response = Mock(message_ids=["id1", "id2"])
        self.mock_stub.PushMessages.return_value = mock_response
        msgs = [Message(RecordDict(), 0, "query") for _ in range(2)]
        # Use invalid run_id
        msgs[1].metadata.__dict__["_message_id"] = "invalid message id"

        # Execute and assert
        with self.assertRaises(ValueError):
            self.grid.push_messages(msgs)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        mock_response = Mock()
        # A Message must have either content or error set so we prepare
        run_id = 12345
        ok_message = create_res_message(src_node_id=123, dst_node_id=456, run_id=run_id)
        ok_message.metadata.reply_to_message_id = "id2"

        error_message = create_res_message(
            src_node_id=123, dst_node_id=789, run_id=run_id, error=Error(code=0)
        )
        error_message.metadata.reply_to_message_id = "id3"
        # The response from the ServerAppIoServicer is in the form of Protobuf Messages
        mock_response.messages_list = [ok_message, error_message]
        self.mock_stub.PullMessages.return_value = mock_response
        msg_ids = ["id1", "id2", "id3"]

        # Execute
        msgs = self.grid.pull_messages(msg_ids)
        reply_tos = {msg.metadata.reply_to_message_id for msg in msgs}
        args, kwargs = self.mock_stub.PullMessages.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PullResMessagesRequest)
        self.assertEqual(args[0].message_ids, msg_ids)
        self.assertEqual(reply_tos, {"id2", "id3"})

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        mock_response = Mock(message_ids=["id1"])
        self.mock_stub.PushMessages.return_value = mock_response
        # The response message must include either `content` (i.e. a recorddict) or
        # an `Error`. We choose the latter in this case
        run_id = 1234
        mssg = create_res_message(
            src_node_id=123, dst_node_id=456, run_id=run_id, error=Error(code=0)
        )
        mssg.metadata.reply_to_message_id = "id1"
        message_res_list = [mssg]

        mock_response.messages_list = message_res_list
        self.mock_stub.PullMessages.return_value = mock_response
        msgs = [Message(RecordDict(), 0, "query")]

        # Execute
        ret_msgs = list(self.grid.send_and_receive(msgs))

        # Assert
        self.assertEqual(len(ret_msgs), 1)
        self.assertEqual(ret_msgs[0].metadata.reply_to_message_id, "id1")

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        sleep_fn = time.sleep
        mock_response = Mock(message_ids=["id1"])
        self.mock_stub.PushMessages.return_value = mock_response
        mock_response = Mock(messages_list=[])
        self.mock_stub.PullMessages.return_value = mock_response
        msgs = [Message(RecordDict(), 0, "query")]

        # Execute
        with patch("time.sleep", side_effect=lambda t: sleep_fn(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.grid.send_and_receive(msgs, timeout=0.15))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_del_with_initialized_grid(self) -> None:
        """Test cleanup behavior when Grid is initialized."""
        # Execute
        self.grid.close()

        # Assert
        self.mock_channel.close.assert_called_once()

    def test_del_with_uninitialized_grid(self) -> None:
        """Test cleanup behavior when Grid is not initialized."""
        # Prepare
        self.grid._grpc_stub = None  # pylint: disable=protected-access
        self.grid._channel = None  # pylint: disable=protected-access

        # Execute
        self.grid.close()

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
        self.grid._grpc_stub = Mock(
            GetNodes=lambda *args, **kwargs: self.grid._retry_invoker.invoke(
                mock_get_nodes, *args, **kwargs
            )
        )
        # pylint: enable=protected-access

        # Execute
        with patch("time.sleep", side_effect=lambda _: None):
            node_ids = self.grid.get_node_ids()

        # Assert
        self.assertIn(404, node_ids)
        self.assertEqual(mock_get_nodes.call_count, 2)
