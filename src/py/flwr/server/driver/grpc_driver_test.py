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
from logging import WARNING
from unittest.mock import Mock, patch

import grpc

from flwr.common import DEFAULT_TTL, Message, RecordSet
from flwr.common.message import Error
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
        mock_response = Mock(message_ids=["id1", "id2"])
        self.mock_stub.PushMessages.return_value = mock_response
        msgs = [
            self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)
            for _ in range(2)
        ]

        # Execute
        msg_ids = self.driver.push_messages(msgs)
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
        msgs = [
            self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)
            for _ in range(2)
        ]
        # Use invalid run_id
        msgs[1].metadata.__dict__["_run_id"] += 1  # pylint: disable=protected-access

        # Execute and assert
        with self.assertRaises(ValueError):
            self.driver.push_messages(msgs)

    def _setup_pull_messages_mocks(self, run_id: int) -> list[str]:
        """Set up common mocks for pull_messages.

        This creates two mock responses, one for each message ID, and configures the
        stub to return them sequentially.
        """
        # Create messages with distinct reply_to values
        ok_message = create_res_message(src_node_id=123, dst_node_id=456, run_id=run_id)
        ok_message.metadata.reply_to_message = "id2"
        error_message = create_res_message(
            src_node_id=123, dst_node_id=789, run_id=run_id, error=Error(code=0)
        )
        error_message.metadata.reply_to_message = "id3"

        # Create separate mock responses for each call
        mock_response1 = Mock()
        mock_response1.messages_list = [ok_message]
        mock_response2 = Mock()
        mock_response2.messages_list = [error_message]

        # Configure PullMessages to return a different response per call
        self.mock_stub.PullMessages.side_effect = [mock_response1, mock_response2]

        # Return the message IDs to be used in the tests (the valid ones)
        return ["id2", "id3"]

    def _assert_pull_messages(
        self, expected_ids: list[str], messages: list[Message]
    ) -> None:
        """Check that PullMessages was called once per expected message ID and that the
        returned messages have the correct reply_to values."""
        reply_tos = {msg.metadata.reply_to_message for msg in messages}
        calls = self.mock_stub.PullMessages.call_args_list
        self.assertEqual(len(calls), len(expected_ids))
        for call, expected_msg_id in zip(calls, expected_ids):
            args, kwargs = call
            self.assertEqual(len(args), 1)
            self.assertEqual(len(kwargs), 0)
            self.assertIsInstance(args[0], PullResMessagesRequest)
            # Each call should be made with a single-element list
            # containing the expected message id
            self.assertEqual(args[0].message_ids, [expected_msg_id])
        self.assertEqual(reply_tos, {"id2", "id3"})

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        run_id = 12345
        msg_ids = self._setup_pull_messages_mocks(run_id)

        # Store message IDs in the driver's internal state for testing
        self.driver._message_ids.extend(msg_ids)  # pylint: disable=protected-access

        # Execute
        messages = list(self.driver.pull_messages(msg_ids))

        # Assert
        self._assert_pull_messages(msg_ids, messages)

    def test_pull_messages_without_given_message_ids(self) -> None:
        """Test pulling messages successful when no message_ids are provided."""
        # Prepare
        run_id = 12345
        msg_ids = self._setup_pull_messages_mocks(run_id)

        # Store message IDs in the driver's internal state for testing
        self.driver._message_ids.extend(msg_ids)  # pylint: disable=protected-access

        # Execute
        messages = list(self.driver.pull_messages())

        # Assert
        self._assert_pull_messages(msg_ids, messages)

    def test_pull_messages_with_invalid_message_ids(self) -> None:
        """Test pulling messages when provided message_ids include values not stored in
        self._message_ids."""
        # Prepare
        run_id = 12345
        valid_msg_ids = self._setup_pull_messages_mocks(
            run_id
        )  # returns ["id2", "id3"]
        # Store message IDs in the driver's internal state for testing
        self.driver._message_ids.extend(  # pylint: disable=protected-access
            valid_msg_ids
        )
        provided_msg_ids = [
            "id2",
            "id3",
            "id4",
            "id5",
        ]  # "id4" and "id5" are not stored.
        expected_missing = [
            msg_id for msg_id in provided_msg_ids if msg_id not in valid_msg_ids
        ]

        # Patch the log function to capture the warning.
        with patch("flwr.server.driver.grpc_driver.log") as mock_log:
            # Execute
            messages = list(self.driver.pull_messages(provided_msg_ids))

        # Assert
        # Only valid IDs are pulled
        self._assert_pull_messages(valid_msg_ids, messages)
        # Warning was logged with the missing IDs
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        log_level = args[0]
        logged_missing_ids = args[2]
        # Verify that the log level is WARNING and the missing IDs appear in the
        # log message
        self.assertEqual(log_level, WARNING)
        self.assertEqual(logged_missing_ids, expected_missing)

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        mock_response = Mock(message_ids=["id1"])
        self.mock_stub.PushMessages.return_value = mock_response
        # The response message must include either `content` (i.e. a recordset) or
        # an `Error`. We choose the latter in this case
        run_id = 1234
        mssg = create_res_message(
            src_node_id=123, dst_node_id=456, run_id=run_id, error=Error(code=0)
        )
        mssg.metadata.reply_to_message = "id1"
        message_res_list = [mssg]

        mock_response.messages_list = message_res_list
        self.mock_stub.PullMessages.return_value = mock_response
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
        mock_response = Mock(message_ids=["id1"])
        self.mock_stub.PushMessages.return_value = mock_response
        msgs = [self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)]

        # Execute
        # Patch pull_messages to always return an empty iterator.
        with patch.object(self.driver, "pull_messages", return_value=iter([])):
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
