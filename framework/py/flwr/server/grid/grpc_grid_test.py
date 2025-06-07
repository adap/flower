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

from flwr.app.error import Error
from flwr.common import RecordDict
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.inflatable import get_all_nested_objects
from flwr.common.message import Message
from flwr.common.serde import message_to_proto
from flwr.proto.message_pb2 import ObjectIDs  # pylint: disable=E0611
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

    def _prep_message(self, message: Message) -> Message:
        # We need to be able to specify the actual object IDs
        # in the mocked responses, due to this we need to set
        # elements in the metadata that would be normally be
        # set when pushing a message.
        # pylint: disable-next=W0212
        message.metadata._run_id = 61016  # type: ignore
        # pylint: disable-next=W0212
        message.metadata._src_node_id = SUPERLINK_NODE_ID  # type: ignore
        message.metadata.__dict__["_message_id"] = message.object_id
        return message

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        msg1 = self._prep_message(Message(RecordDict(), 0, "query.A"))
        msg2 = self._prep_message(Message(RecordDict(), 0, "query.B"))

        msgs = [msg1, msg2]
        mock_response_msg1 = Mock(
            message_ids=[msg1.object_id],
            objects_to_push={
                msg1.object_id: ObjectIDs(
                    object_ids=[msg1.object_id, RecordDict().object_id]
                ),
            },
        )
        mock_response_msg2 = Mock(
            message_ids=[msg2.object_id],
            objects_to_push={
                msg2.object_id: ObjectIDs(
                    object_ids=[msg2.object_id, RecordDict().object_id]
                ),
            },
        )
        self.mock_stub.PushMessages.side_effect = [
            mock_response_msg1,
            mock_response_msg2,
        ]
        self.mock_stub.PushObject.return_value = Mock(stored=True)

        # Execute
        msg_ids = self.grid.push_messages(msgs)
        args, kwargs = self.mock_stub.PushMessages.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PushInsMessagesRequest)
        self.assertEqual(msg_ids, [msg1.object_id, msg2.object_id])
        for message in args[0].messages_list:
            self.assertEqual(message.metadata.run_id, 61016)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare: Create instruction messages
        ins1 = self._prep_message(Message(RecordDict(), 123, "query"))
        ins2 = self._prep_message(Message(RecordDict(), 456, "query"))

        # Prepare: Create a normal reply
        ok_msg = Message(RecordDict(), reply_to=ins1)
        ok_msg.metadata.__dict__["_message_id"] = ok_msg.object_id
        ok_msg_all_objs = get_all_nested_objects(ok_msg)
        ok_msg_descendant_ids = set(ok_msg_all_objs.keys()) - {ok_msg.object_id}

        # Prepare: Create an error reply
        err_msg = Message(Error(0), reply_to=ins2)
        err_msg.metadata.__dict__["_message_id"] = err_msg.object_id
        err_msg_all_objs = get_all_nested_objects(err_msg)
        error_msg_descendant_ids = set(err_msg_all_objs.keys()) - {err_msg.object_id}

        # Prepare: Mock the objectStore
        obj_store = {k: v.deflate() for k, v in ok_msg_all_objs.items()}
        obj_store.update({k: v.deflate() for k, v in err_msg_all_objs.items()})

        # Prepare: Mock the response of PushMessages
        self.mock_stub.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(ok_msg), message_to_proto(err_msg)],
            objects_to_pull={
                ok_msg.object_id: Mock(object_ids=ok_msg_descendant_ids),
                err_msg.object_id: Mock(object_ids=error_msg_descendant_ids),
            },
        )

        # Prepare: Mock response of PullObject
        self.mock_stub.PullObject.side_effect = lambda req: Mock(
            object_found=True,
            object_available=True,
            object_content=obj_store[req.object_id],
        )

        # Execute
        msgs = list(self.grid.pull_messages([ins1.object_id, ins2.object_id]))
        args, kwargs = self.mock_stub.PullMessages.call_args

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PullResMessagesRequest)
        self.assertEqual(args[0].message_ids, [ins1.object_id, ins2.object_id])
        self.assertEqual(msgs[0].metadata, ok_msg.metadata)
        self.assertEqual(msgs[0].content, ok_msg.content)
        self.assertEqual(msgs[1].metadata, err_msg.metadata)
        self.assertEqual(msgs[1].error, err_msg.error)
        self.assertEqual(self.mock_stub.PullObject.call_count, len(obj_store))

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare: Create an instruction message and mock responses
        msg = self._prep_message(Message(RecordDict(), 0, "query"))
        self.mock_stub.PushMessages.return_value = Mock(
            message_ids=[msg.object_id],
            objects_to_push={
                msg.object_id: ObjectIDs(
                    object_ids=[msg.object_id, RecordDict().object_id]
                ),
            },
        )
        self.mock_stub.PushObject.return_value = Mock(stored=True)

        # Prepare: create an error reply message and mock responses
        reply = Message(Error(0), reply_to=msg)
        reply.metadata.__dict__["_message_id"] = reply.object_id
        self.mock_stub.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(reply)],
            objects_to_pull={
                reply.object_id: Mock(object_ids=[reply.object_id]),
            },
        )
        self.mock_stub.PullObject.return_value = Mock(
            object_found=True, object_available=True, object_content=reply.deflate()
        )

        # Execute
        ret_msgs = list(self.grid.send_and_receive([msg]))

        # Assert
        self.assertEqual(len(ret_msgs), 1)
        self.assertEqual(ret_msgs[0].metadata, reply.metadata)
        self.assertEqual(ret_msgs[0].error, reply.error)

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        msg = self._prep_message(Message(RecordDict(), 0, "query"))
        sleep_fn = time.sleep
        mock_response = Mock(
            message_ids=[msg.object_id],
            objects_to_push={
                msg.object_id: ObjectIDs(
                    object_ids=[msg.object_id, RecordDict().object_id]
                ),
            },
        )
        self.mock_stub.PushMessages.return_value = mock_response
        self.mock_stub.PushObject.return_value = Mock(stored=True)
        mock_response = Mock(messages_list=[])
        self.mock_stub.PullMessages.return_value = mock_response

        # Execute
        with patch("time.sleep", side_effect=lambda t: sleep_fn(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.grid.send_and_receive([msg], timeout=0.15))

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
