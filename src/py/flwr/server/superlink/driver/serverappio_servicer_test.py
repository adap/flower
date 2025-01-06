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
"""ServerAppIoServicer tests."""


import tempfile
import unittest

import grpc
from parameterized import parameterized

from flwr.common import ConfigsRecord, Context
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS, Status
from flwr.common.serde import context_to_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import RunStatus
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushServerAppOutputsRequest,
    PushServerAppOutputsResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611
from flwr.server.superlink.driver.serverappio_grpc import run_serverappio_api_grpc
from flwr.server.superlink.driver.serverappio_servicer import _raise_if
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_task_ins
from flwr.server.superlink.utils import _STATUS_TO_MSG

# pylint: disable=broad-except


def test_raise_if_false() -> None:
    """."""
    # Prepare
    validation_error = False
    detail = "test"

    try:
        # Execute
        _raise_if(validation_error, detail)

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
        _raise_if(validation_error, detail)

        # Assert
        raise AssertionError()
    except ValueError as err:
        assert str(err) == "Malformed PushTaskInsRequest: test"
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
        self._push_task_ins = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushTaskIns",
            request_serializer=PushTaskInsRequest.SerializeToString,
            response_deserializer=PushTaskInsResponse.FromString,
        )
        self._pull_task_res = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullTaskRes",
            request_serializer=PullTaskResRequest.SerializeToString,
            response_deserializer=PullTaskResResponse.FromString,
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())

        # Transition status to running. PushTaskRes is only allowed in running status.
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_nodes_not_allowed(run_id)

    def test_successful_push_task_ins_if_running(self) -> None:
        """Test `PushTaskIns` success."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        # Transition status to running. PushTaskRes is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushTaskInsRequest(task_ins_list=[task_ins], run_id=run_id)

        # Execute
        response, call = self._push_task_ins.with_call(request=request)

        # Assert
        assert isinstance(response, PushTaskInsResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_push_task_ins_not_allowed(self, task_ins: TaskIns, run_id: int) -> None:
        """Assert `PushTaskIns` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushTaskInsRequest(task_ins_list=[task_ins], run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._push_task_ins.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_task_ins_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushTaskIns` not successful if RunStatus is not running."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_task_ins_not_allowed(task_ins, run_id)

    def test_pull_task_res_successful_if_running(self) -> None:
        """Test `PullTaskRes` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        # Transition status to running. PushTaskRes is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PullTaskResRequest(task_ids=[], run_id=run_id)

        # Execute
        response, call = self._pull_task_res.with_call(request=request)

        # Assert
        assert isinstance(response, PullTaskResResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_pull_task_res_not_allowed(self, run_id: int) -> None:
        """Assert `PullTaskRes` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PullTaskResRequest(node=Node(node_id=0), run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._pull_task_res.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_pull_task_res_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PullTaskRes` not successful if RunStatus is not running."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_pull_task_res_not_allowed(run_id)

    def test_push_serverapp_outputs_successful_if_running(self) -> None:
        """Test `PushServerAppOutputs` success."""
        # Prepare
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recordset(1, 1, 1),
            run_config=maker.user_config(),
        )

        # Transition status to running. PushTaskRes is only allowed in running status.
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recordset(1, 1, 1),
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
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
