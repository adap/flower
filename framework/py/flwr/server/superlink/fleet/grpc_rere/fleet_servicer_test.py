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
"""Flower FleetServicer tests."""


import tempfile
import unittest

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, Message
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.inflatable import get_descendant_object_ids, get_object_id
from flwr.common.message import get_message_to_descendant_id_mapping
from flwr.common.serde import message_from_proto
from flwr.common.typing import RunStatus
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.app import _run_fleet_api_grpc_rere
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import (
    create_ins_message,
    create_res_message,
)
from flwr.server.superlink.utils import _STATUS_TO_MSG
from flwr.supercore.object_store import ObjectStoreFactory


class TestFleetServicer(unittest.TestCase):  # pylint: disable=R0902
    """FleetServicer tests for allowed RunStatuses."""

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

        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            ffs_factory,
            objectstore_factory,
            None,
            None,
        )

        self._channel = grpc.insecure_channel("localhost:9092")
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushMessages",
            request_serializer=PushMessagesRequest.SerializeToString,
            response_deserializer=PushMessagesResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullMessages",
            request_serializer=PullMessagesRequest.SerializeToString,
            response_deserializer=PullMessagesResponse.FromString,
        )
        self._get_run = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetRun",
            request_serializer=GetRunRequest.SerializeToString,
            response_deserializer=GetRunResponse.FromString,
        )
        self._get_fab = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetFab",
            request_serializer=GetFabRequest.SerializeToString,
            response_deserializer=GetFabResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
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

    def test_successful_push_messages_if_running(self) -> None:
        """Test `PushMessages` success."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Transition status to running. PushMessages RPC is only allowed in
        # running status.
        self._transition_run_status(run_id, 2)

        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )

        # Construct message to descendant mapping
        message = message_from_proto(msg_proto)
        descendant_mapping = get_message_to_descendant_id_mapping(message)

        request = PushMessagesRequest(
            node=Node(node_id=node_id),
            messages_list=[msg_proto],
            msg_to_descendant_mapping=descendant_mapping,
        )

        # Execute
        response, call = self._push_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PushMessagesResponse)
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

    def _assert_push_messages_not_allowed(self, node_id: int, run_id: int) -> None:
        """Assert `PushMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]

        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )
        request = PushMessagesRequest(
            node=Node(node_id=node_id), messages_list=[msg_proto]
        )

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
    def test_push_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushMessages` not successful if RunStatus is not running."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_messages_not_allowed(node_id, run_id)

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
        obj_ids_registered = self.store.preregister(descendants + [message_obj_id])

        return obj_ids_registered

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )  # type: ignore
    def test_pull_messages_if_running(
        self,
        register_in_store: bool,
    ) -> None:
        """Test `PullMessages` success if objects are registered in ObjectStore."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)

        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Transition status to running. PullMessagesRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)

        # Let's insert a Message in the LinkState and register it in the ObjectStore
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore
        self.state.store_message_ins(message=message_ins)
        obj_ids_registered: list[str] = []
        if register_in_store:
            obj_ids_registered = self._register_in_object_store(message_ins)

        request = PullMessagesRequest(node=Node(node_id=node_id))

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullMessagesResponse)
        assert call.code() == grpc.StatusCode.OK

        object_ids_in_response = {
            obj_id
            for obj_ids in response.objects_to_pull.values()
            for obj_id in obj_ids.object_ids
        }
        if register_in_store:
            # Assert expected object_ids
            assert set(obj_ids_registered) == object_ids_in_response
            assert message_ins.object_id == list(response.objects_to_pull.keys())[0]
        else:
            assert set() == object_ids_in_response
            # Ins message was deleted
            assert self.state.num_message_ins() == 0

    def test_successful_get_run_if_running(self) -> None:
        """Test `GetRun` success."""
        # Prepare
        self.state.create_node(heartbeat_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Transition status to running. GetRun RPC is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = GetRunRequest(run_id=run_id)

        # Execute
        response, call = self._get_run.with_call(request=request)

        # Assert
        assert isinstance(response, GetRunResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_run_not_allowed(self, run_id: int) -> None:
        """Assert `GetRun` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetRunRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._get_run.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_run_not_successful_if_not_running(self, num_transitions: int) -> None:
        """Test `GetRun` not successful if RunStatus is not running."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_run_not_allowed(run_id)

    def test_successful_get_fab_if_running(self) -> None:
        """Test `GetFab` success."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        fab_content = b"content"
        fab_hash = self.ffs.put(fab_content, {"meta": "data"})
        run_id = self.state.create_run("", "", fab_hash, {}, ConfigRecord())

        # Transition status to running. GetFab RPC is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=fab_hash, run_id=run_id
        )

        # Execute
        response, call = self._get_fab.with_call(request=request)

        # Assert
        assert isinstance(response, GetFabResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_fab_not_allowed(
        self, node_id: int, hash_str: str, run_id: int
    ) -> None:
        """Assert `GetFab` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=hash_str, run_id=run_id
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._get_fab.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_fab_not_successful_if_not_running(self, num_transitions: int) -> None:
        """Test `GetFab` not successful if RunStatus is not running."""
        # Prepare
        node_id = self.state.create_node(heartbeat_interval=30)
        fab_content = b"content"
        fab_hash = self.ffs.put(fab_content, {"meta": "data"})
        run_id = self.state.create_run("", "", fab_hash, {}, ConfigRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_fab_not_allowed(node_id, fab_hash, run_id)

    def test_push_object_succesful(self) -> None:
        """Test `PushObject`."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        node_id = self.state.create_node(heartbeat_interval=30)
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()
        self._transition_run_status(run_id, 2)

        # Pre-register object
        self.store.preregister(object_ids=[obj.object_id])

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Empty response
        assert res.stored

    def test_push_object_fails(self) -> None:
        """Test `PushObject` in unsupported scenarios."""
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Run is not running
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._push_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Prepare
        self._transition_run_status(run_id, 2)
        node_id = self.state.create_node(heartbeat_interval=30)
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Push valid object but it hasn't been pre-registered
        req = PushObjectRequest(
            node=Node(node_id=node_id),
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
        self.store.preregister(object_ids=[fake_object_id])

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=node_id),
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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        self._transition_run_status(run_id, 2)
        node_id = self.state.create_node(heartbeat_interval=30)
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Preregister object
        self.store.preregister(object_ids=[obj.object_id])

        # Pull
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id=obj.object_id
        )
        res: PullObjectResponse = self._pull_object(req)

        # Assert object content is b"" (it was never pushed)
        assert res.object_found
        assert not res.object_available
        assert res.object_content == b""

        # Put object in store, then check it can be pulled
        self.store.put(object_id=obj.object_id, object_content=obj_b)
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id=obj.object_id
        )
        res = self._pull_object(req)

        # Assert, identical object pulled
        assert res.object_found
        assert res.object_available
        assert obj_b == res.object_content

    def test_pull_object_fails(self) -> None:
        """Test `PullObject` in unsuported scenarios."""
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Run is not running
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Attempt pulling object that doesn't exist
        self._transition_run_status(run_id, 2)
        node_id = self.state.create_node(heartbeat_interval=30)
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id="1234"
        )
        res: PullObjectResponse = self._pull_object(req)
        # Empty response
        assert not res.object_found
