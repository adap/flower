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
"""ServerAppIoServicer tests."""


import tempfile
import unittest
from typing import Optional
from unittest.mock import patch

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Error, Message, RecordDict
from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.inflatable import (
    get_all_nested_objects,
    get_descendant_object_ids,
    get_object_id,
    get_object_tree,
)
from flwr.common.message import get_message_to_descendant_id_mapping
from flwr.common.serde import context_to_proto, message_from_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import RunStatus
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
)
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ObjectTree,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
    PullResMessagesRequest,
    PullResMessagesResponse,
    PushInsMessagesRequest,
    PushInsMessagesResponse,
    PushServerAppOutputsRequest,
    PushServerAppOutputsResponse,
)
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.server.superlink.serverappio.serverappio_servicer import _raise_if
from flwr.server.superlink.utils import _STATUS_TO_MSG
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory

# pylint: disable=broad-except


def test_raise_if_false() -> None:
    """."""
    # Prepare
    validation_error = False
    detail = "test"

    try:
        # Execute
        _raise_if(
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        assert True
    except ValueError as err:
        raise AssertionError() from err
    except Exception as err:
        raise AssertionError() from err


def test_raise_if_true() -> None:
    """."""
    # Prepare
    validation_error = True
    detail = "test"

    try:
        # Execute
        _raise_if(
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        raise AssertionError()
    except ValueError as err:
        assert str(err) == "Malformed DummyRequest: test"
    except Exception as err:
        raise AssertionError() from err


class TestServerAppIoServicer(unittest.TestCase):  # pylint: disable=R0902, R0904
    """ServerAppIoServicer tests for allowed RunStatuses."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)  # Ensures cleanup after test

        state_factory = LinkStateFactory(":flwr-in-memory-state:")
        self.state = state_factory.state()
        ffs_factory = FfsFactory(self.temp_dir.name)
        self.ffs = ffs_factory.ffs()
        objectstore_factory = ObjectStoreFactory()
        self.store = objectstore_factory.store()

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            ffs_factory,
            objectstore_factory,
            None,
        )

        self._channel = grpc.insecure_channel("localhost:9091")
        self._get_nodes = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushMessages",
            request_serializer=PushInsMessagesRequest.SerializeToString,
            response_deserializer=PushInsMessagesResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullMessages",
            request_serializer=PullResMessagesRequest.SerializeToString,
            response_deserializer=PullResMessagesResponse.FromString,
        )
        self._push_serverapp_outputs = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushServerAppOutputs",
            request_serializer=PushServerAppOutputsRequest.SerializeToString,
            response_deserializer=PushServerAppOutputsResponse.FromString,
        )
        self._update_run_status = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/UpdateRunStatus",
            request_serializer=UpdateRunStatusRequest.SerializeToString,
            response_deserializer=UpdateRunStatusResponse.FromString,
        )
        self._send_app_heartbeat = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/SendAppHeartbeat",
            request_serializer=SendAppHeartbeatRequest.SerializeToString,
            response_deserializer=SendAppHeartbeatResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._confirm_message_received = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/ConfirmMessageReceived",
            request_serializer=ConfirmMessageReceivedRequest.SerializeToString,
            response_deserializer=ConfirmMessageReceivedResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def _transition_run_status(self, run_id: int, num_transitions: int) -> None:
        if num_transitions > 0:
            _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        if num_transitions > 1:
            _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        if num_transitions > 2:
            _ = self.state.update_run_status(run_id, RunStatus(Status.FINISHED, "", ""))

    def test_successful_get_node_if_running(self) -> None:
        """Test `GetNode` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        # Transition status to running. GetNodesRequest is only allowed
        # in running status.
        self._transition_run_status(run_id, 2)
        request = GetNodesRequest(run_id=run_id)

        # Execute
        response, call = self._get_nodes.with_call(request=request)

        # Assert
        assert isinstance(response, GetNodesResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_nodes_not_allowed(self, run_id: int) -> None:
        """Assert `GetNodes` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetNodesRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._get_nodes.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_nodes_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `GetNodes` not sucessful if RunStatus is pending."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_nodes_not_allowed(run_id)

    def test_successful_push_messages_if_running(self) -> None:
        """Test `PushMessages` success."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )

        # Construct message to descendant mapping
        message = message_from_proto(message_ins)
        descendant_mapping = get_message_to_descendant_id_mapping(message)

        # Transition status to running. PushInsMessagesRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushInsMessagesRequest(
            messages_list=[message_ins],
            run_id=run_id,
            message_object_trees=[get_object_tree(message)],
        )

        # Execute
        response, call = self._push_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PushInsMessagesResponse)
        assert grpc.StatusCode.OK == call.code()

        # Assert: check that response indicates all objects need pushing
        expected_object_ids = {message.object_id}  # message
        expected_object_ids |= {
            obj_id
            for obj_ids in descendant_mapping.values()
            for obj_id in obj_ids.object_ids
        }  # descendants
        # Construct a single set with all object ids
        requested_object_ids = {
            obj_id
            for obj_ids in response.objects_to_push.values()
            for obj_id in obj_ids.object_ids
        }
        assert expected_object_ids == requested_object_ids
        assert response.objects_to_push.keys() == descendant_mapping.keys()

    def _assert_push_ins_messages_not_allowed(
        self, message: ProtoMessage, run_id: int
    ) -> None:
        """Assert `PushInsMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushInsMessagesRequest(messages_list=[message], run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._push_messages.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_ins_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushInsMessages` not successful if RunStatus is not running."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_ins_messages_not_allowed(message_ins, run_id)

    def _register_in_object_store(self, message: Message) -> list[str]:
        # When pulling a Message, the response also must include the IDs of the objects
        # to pull. To achieve this, we need to at least register the Objects in the
        # message into the store. Note this would normally be done when the
        # servicer handles a PushMessageRequest
        descendants = list(get_descendant_object_ids(message))
        message_obj_id = message.metadata.message_id
        # Store mapping
        self.store.set_message_descendant_ids(
            msg_object_id=message_obj_id, descendant_ids=descendants
        )
        # Preregister
        obj_ids_registered = self.store.preregister(
            message.metadata.run_id, get_object_tree(message)
        )

        return obj_ids_registered

    @parameterized.expand(
        [
            # The normal case:
            # The message is recognized by both `LinkState` and `ObjectStore`
            (True,),
            # The failure case:
            # The message is found in `LinkState` but not in `ObjectStore`
            (False,),
        ]
    )  # type: ignore
    def test_pull_messages_if_running(self, register_in_store: bool) -> None:
        """Test `PullMessages` success if objects are registered in ObjectStore."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        node_id = self.state.create_node(heartbeat_interval=30)
        # Transition status to running. PullResMessagesRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore
        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=node_id, limit=1)[0]

        reply_msg = Message(RecordDict(), reply_to=msg_)
        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore
        self.state.store_message_res(message=reply_msg)

        # Register response in ObjectStore (so pulling message request can be completed)
        obj_ids_registered: list[str] = []
        if register_in_store:
            obj_ids_registered = self._register_in_object_store(reply_msg)

        request = PullResMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullResMessagesResponse)
        assert call.code() == grpc.StatusCode.OK

        object_ids_in_response = {
            obj_id
            for obj_ids in response.objects_to_pull.values()
            for obj_id in obj_ids.object_ids
        }
        object_ids_in_response |= set(response.objects_to_pull.keys())
        if register_in_store:
            # Assert expected object_ids
            assert set(obj_ids_registered) == object_ids_in_response
            assert reply_msg.object_id == list(response.objects_to_pull.keys())[0]
        else:
            assert set() == object_ids_in_response
            # Ins message was deleted
            assert self.state.num_message_ins() == 0

    @parameterized.expand(
        [
            # Reply with Message
            (RecordDict(), None),
            # Reply with Error
            (None, Error(code=0)),
        ]
    )  # type: ignore
    def test_successful_pull_messages_deletes_messages_in_linkstate(
        self, content: Optional[RecordDict], error: Optional[Error]
    ) -> None:
        """Test `PullMessages` deletes messages from LinkState."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        # Transition status to running.
        self._transition_run_status(run_id, 2)

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore

        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=node_id, limit=1)[0]

        if content is not None:
            reply_msg = Message(content, reply_to=msg_)
        else:
            assert error is not None
            reply_msg = Message(error, reply_to=msg_)

        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore

        self.state.store_message_res(message=reply_msg)
        # Register response in ObjectStore (so pulling message request can be completed)
        self._register_in_object_store(reply_msg)
        request = PullResMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullResMessagesResponse)
        assert grpc.StatusCode.OK == call.code()
        assert self.state.num_message_ins() == 0
        assert self.state.num_message_res() == 0

    def _assert_pull_messages_not_allowed(self, run_id: int) -> None:
        """Assert `PullMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PullResMessagesRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._pull_messages.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_pull_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PullMessages` not successful if RunStatus is not running."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_pull_messages_not_allowed(run_id)

    def test_pull_message_from_expired_message_error(self) -> None:
        """Test that the servicer correctly handles the registration in the ObjectStore
        of an Error message created by the LinkState due to an expired TTL."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        # Transition status to running.
        self._transition_run_status(run_id, 2)

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        msg_id = self.state.store_message_ins(message=message_ins)

        # Simulate situation where the message has expired in the LinkState
        # This will trigger the creation of an Error message
        with patch(
            "time.time",
            side_effect=lambda: message_ins.metadata.created_at
            + message_ins.metadata.ttl
            + 0.1,
        ):  # over TTL limit

            request = PullResMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)

            # Execute
            response, call = self._pull_messages.with_call(request=request)

            # Assert
            assert isinstance(response, PullResMessagesResponse)
            assert grpc.StatusCode.OK == call.code()

            # Assert that objects to pull points to a message carrying an error
            msg_res = message_from_proto(response.messages_list[0])
            assert msg_res.has_error()
            # objects_to_pull is expected to be {msg_obj_id: []}
            assert list(response.objects_to_pull.keys()) == [msg_res.object_id]
            assert list(response.objects_to_pull.values())[0].object_ids == []

    def test_push_serverapp_outputs_successful_if_running(self) -> None:
        """Test `PushServerAppOutputs` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recorddict(1, 1, 1),
            run_config=maker.user_config(),
        )

        # Transition status to running. PushServerAppOutputsRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushServerAppOutputsRequest(
            run_id=run_id, context=context_to_proto(context)
        )

        # Execute
        response, call = self._push_serverapp_outputs.with_call(request=request)

        # Assert
        assert isinstance(response, PushServerAppOutputsResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_push_serverapp_outputs_not_allowed(
        self, run_id: int, context: Context
    ) -> None:
        """Assert `PushServerAppOutputs` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushServerAppOutputsRequest(
            run_id=run_id, context=context_to_proto(context)
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._push_serverapp_outputs.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_serverapp_outputs_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushServerAppOutputs` not successful if RunStatus is not running."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recorddict(1, 1, 1),
            run_config=maker.user_config(),
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_serverapp_outputs_not_allowed(run_id, context)

    @parameterized.expand(
        [
            (0,),  # Test successful if RunStatus is pending.
            (1,),  # Test successful if RunStatus is starting.
            (2,),  # Test successful if RunStatus is running.
        ]
    )  # type: ignore
    def test_update_run_status_successful_if_not_finished(
        self, num_transitions: int
    ) -> None:
        """Test `UpdateRunStatus` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        _ = self.state.get_run_status({run_id})[run_id]
        next_run_status = RunStatus(Status.STARTING, "", "")

        if num_transitions > 0:
            _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
            next_run_status = RunStatus(Status.RUNNING, "", "")
        if num_transitions > 1:
            _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
            next_run_status = RunStatus(Status.FINISHED, "", "")

        request = UpdateRunStatusRequest(
            run_id=run_id, run_status=run_status_to_proto(next_run_status)
        )

        # Execute
        response, call = self._update_run_status.with_call(request=request)

        # Assert
        assert isinstance(response, UpdateRunStatusResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_update_run_status_not_successful_if_finished(self) -> None:
        """Test `UpdateRunStatus` not successful."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        _ = self.state.get_run_status({run_id})[run_id]
        _ = self.state.update_run_status(run_id, RunStatus(Status.FINISHED, "", ""))
        run_status = self.state.get_run_status({run_id})[run_id]
        next_run_status = RunStatus(Status.FINISHED, "", "")

        request = UpdateRunStatusRequest(
            run_id=run_id, run_status=run_status_to_proto(next_run_status)
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._update_run_status.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand([(1,), (2,)])  # type: ignore
    def test_successful_send_app_heartbeat(self, num_transitions: int) -> None:
        """Test `SendAppHeartbeat` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to starting or running.
        self._transition_run_status(run_id, num_transitions)
        request = SendAppHeartbeatRequest(run_id=run_id, heartbeat_interval=30)

        # Execute
        response, call = self._send_app_heartbeat.with_call(request=request)

        # Assert
        assert isinstance(response, SendAppHeartbeatResponse)
        assert grpc.StatusCode.OK == call.code()
        assert response.success

    @parameterized.expand([(0,), (3,)])  # type: ignore
    def test_send_app_heartbeat_not_successful(self, num_transitions: int) -> None:
        """Test `SendAppHeartbeat` not successful when status is pending or finished."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Stay in pending or transition to finished
        self._transition_run_status(run_id, num_transitions)
        request = SendAppHeartbeatRequest(run_id=run_id, heartbeat_interval=30)

        # Execute
        response, _ = self._send_app_heartbeat.with_call(request=request)

        # Assert
        assert not response.success

    def test_push_object_succesful(self) -> None:
        """Test `PushObject`."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        self._transition_run_status(run_id, 2)
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Pre-register object
        self.store.preregister(run_id, get_object_tree(obj))

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Empty response
        assert res.stored

    def test_push_object_fails(self) -> None:
        """Test `PushObject` in unsupported scenarios."""
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Run is not running
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._push_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Run is running but node ID isn't recognized
        self._transition_run_status(run_id, 2)
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._push_object(request=req)
        assert e.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

        # Prepare
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Push valid object but it hasn't been pre-registered
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Assert: object not inserted
        assert not res.stored

        # Push valid object but its hash doesnt match the one passed in the request
        # Preregister under a different object-id
        fake_object_id = get_object_id(b"1234")
        self.store.preregister(run_id, ObjectTree(object_id=fake_object_id))

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=fake_object_id,
            object_content=obj_b,
        )
        res = self._push_object(request=req)

        # Assert: object not inserted
        assert not res.stored

    def test_pull_object_successful(self) -> None:
        """Test `PullObject` functionality."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        self._transition_run_status(run_id, 2)
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Preregister object
        self.store.preregister(run_id, get_object_tree(obj))

        # Pull
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res: PullObjectResponse = self._pull_object(req)

        # Assert object content is b"" (it was never pushed)
        assert res.object_found
        assert not res.object_available
        assert res.object_content == b""

        # Put object in store, then check it can be pulled
        self.store.put(object_id=obj.object_id, object_content=obj_b)
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res = self._pull_object(req)

        # Assert, identical object pulled
        assert res.object_found
        assert res.object_available
        assert obj_b == res.object_content

    def test_pull_object_fails(self) -> None:
        """Test `PullObject` in unsuported scenarios."""
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Run is not running
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Run is running but node ID isn't recognized
        self._transition_run_status(run_id, 2)
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

        # Attempt pulling object that doesn't exist
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id="1234"
        )
        res: PullObjectResponse = self._pull_object(req)
        # Empty response
        assert not res.object_found

    def test_confirm_message_received_successful(self) -> None:
        """Test `ConfirmMessageReceived` success."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        self._transition_run_status(run_id, 2)
        proto = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )
        message_ins = message_from_proto(proto)
        message_res = Message(
            RecordDict({"cfg": ConfigRecord({"key": "value"})}), reply_to=message_ins
        )

        # Prepare: Save reply message in ObjectStore
        all_objects = get_all_nested_objects(message_res)
        self.store.preregister(run_id, get_object_tree(message_res))
        self.store.set_message_descendant_ids(
            msg_object_id=message_res.object_id,
            descendant_ids=list(get_descendant_object_ids(message_res)),
        )
        for obj_id, obj in all_objects.items():
            self.store.put(object_id=obj_id, object_content=obj.deflate())

        # Assert: All objects are stored in the ObjectStore
        assert len(self.store) == len(all_objects)

        # Execute: Confirm message received
        request = ConfirmMessageReceivedRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            message_object_id=message_res.object_id,
        )
        response, call = self._confirm_message_received.with_call(request=request)

        # Assert
        assert isinstance(response, ConfirmMessageReceivedResponse)
        assert grpc.StatusCode.OK == call.code()

        # Assert: Message is removed from LinkState
        assert len(self.store) == 0
