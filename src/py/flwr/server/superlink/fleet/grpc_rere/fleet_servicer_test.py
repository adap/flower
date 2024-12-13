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
"""Flower FleetServicer tests."""


import tempfile
import unittest

import grpc
from parameterized import parameterized

from flwr.common import ConfigsRecord
from flwr.common.constant import FLEET_API_GRPC_RERE_DEFAULT_ADDRESS, Status
from flwr.common.typing import RunStatus
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskRes  # pylint: disable=E0611
from flwr.server.app import _run_fleet_api_grpc_rere
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.utils import _STATUS_TO_MSG


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

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            ffs_factory,
            None,
            None,
        )

        self._channel = grpc.insecure_channel("localhost:9092")
        self._push_task_res = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushTaskRes",
            request_serializer=PushTaskResRequest.SerializeToString,
            response_deserializer=PushTaskResResponse.FromString,
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

    def test_successful_push_task_res_if_running(self) -> None:
        """Test `PushTaskRes` success."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        # Transition status to running. PushTaskRes is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushTaskResRequest(
            task_res_list=[
                TaskRes(task=Task(producer=Node(node_id=node_id)), run_id=run_id)
            ]
        )

        # Execute
        response, call = self._push_task_res.with_call(request=request)

        # Assert
        assert isinstance(response, PushTaskResResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_push_task_res_not_allowed(self, node_id: int, run_id: int) -> None:
        """Assert `PushTaskRes` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushTaskResRequest(
            task_res_list=[
                TaskRes(task=Task(producer=Node(node_id=node_id)), run_id=run_id)
            ]
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._push_task_res.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_task_res_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushTaskRes` not successful if RunStatus is not running."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_task_res_not_allowed(node_id, run_id)

    def test_successful_get_run_if_running(self) -> None:
        """Test `GetRun` success."""
        # Prepare
        self.state.create_node(ping_interval=30)
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        # Transition status to running. PushTaskRes is only allowed in running status.
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
        run_id = self.state.create_run("", "", "", {}, ConfigsRecord())
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_run_not_allowed(run_id)

    def test_successful_get_fab_if_running(self) -> None:
        """Test `GetFab` success."""
        # Prepare
        node_id = self.state.create_node(ping_interval=30)
        fab_content = b"content"
        fab_hash = self.ffs.put(fab_content, {"meta": "data"})
        run_id = self.state.create_run("", "", fab_hash, {}, ConfigsRecord())

        # Transition status to running. PushTaskRes is only allowed in running status.
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
        node_id = self.state.create_node(ping_interval=30)
        fab_content = b"content"
        fab_hash = self.ffs.put(fab_content, {"meta": "data"})
        run_id = self.state.create_run("", "", fab_hash, {}, ConfigsRecord())

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_fab_not_allowed(node_id, fab_hash, run_id)
