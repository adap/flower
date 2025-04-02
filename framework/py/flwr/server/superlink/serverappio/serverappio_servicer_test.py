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

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Error, Message, RecordDict
from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.serde import context_to_proto, message_from_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import RunStatus
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
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
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.server.superlink.serverappio.serverappio_servicer import _raise_if
from flwr.server.superlink.utils import _STATUS_TO_MSG

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


class TestServerAppIoServicer(unittest.TestCase):  # pylint: disable=R0902
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

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            ffs_factory,
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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_nodes_not_allowed(run_id)

    def test_successful_push_messages_if_running(self) -> None:
        """Test `PushMessages` success."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )

        # Transition status to running. PushInsMessagesRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushInsMessagesRequest(messages_list=[message_ins], run_id=run_id)

        # Execute
        response, call = self._push_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PushInsMessagesResponse)
        assert grpc.StatusCode.OK == call.code()

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
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_ins_messages_not_allowed(message_ins, run_id)

    def test_successful_pull_messages_if_running(self) -> None:
        """Test `PullMessages` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
        # Transition status to running. PullResMessagesRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PullResMessagesRequest(message_ids=[], run_id=run_id)

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullResMessagesResponse)
        assert grpc.StatusCode.OK == call.code()

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
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

        # Transition status to running.
        self._transition_run_status(run_id, 2)

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=node_id, limit=1)[0]

        if content is not None:
            reply_msg = Message(content, reply_to=msg_)
        else:
            assert error is not None
            reply_msg = Message(error, reply_to=msg_)

        self.state.store_message_res(message=reply_msg)

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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_pull_messages_not_allowed(run_id)

    def test_push_serverapp_outputs_successful_if_running(self) -> None:
        """Test `PushServerAppOutputs` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())

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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
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
        run_id = self.state.create_run("", "", "", {}, ConfigRecord())
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
