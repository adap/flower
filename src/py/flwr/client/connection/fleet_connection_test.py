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
"""Tests for all connection implemenations."""


from __future__ import annotations

import unittest
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TypeVar, cast
from unittest.mock import Mock, patch

from google.protobuf.message import Message as GrpcMessage

from flwr.common import Message, serde
from flwr.common.constant import Status
from flwr.common.message import Metadata
from flwr.common.retry_invoker import RetryInvoker, exponential
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import Fab, Run, RunStatus
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611

from . import (
    FleetConnection,
    GrpcAdapterFleetConnection,
    GrpcRereFleetConnection,
    RestFleetConnection,
)

# Tests for GrpcBidiFleetConnections are not included in this file because
# it doesn't support all Fleet APIs.

T = TypeVar("T", bound=GrpcMessage)


mk = RecordMaker()
NODE_ID = 6
FAB_HASH = "03840e932bf61247c1231f0aec9e8ec5f041ed5516fb23638f24d25f3a007acd"
FAB = Fab(FAB_HASH, b"mock fab content")
RUN_ID = 616
RUN_STATUS = RunStatus(Status.PENDING, "", "")
RUN_INFO = Run(
    RUN_ID, "dummy/mock", "v0.0", FAB_HASH, {}, "mock-iso", "", "", "", RUN_STATUS
)
MESSAGE = Message(
    metadata=Metadata(
        run_id=RUN_ID,
        message_id=mk.get_str(64),
        group_id=mk.get_str(30),
        src_node_id=0,
        dst_node_id=NODE_ID,
        reply_to_message="",
        ttl=mk.rng.randint(1, 1 << 30),
        message_type=mk.get_str(10),
    ),
    content=mk.recordset(1, 1, 1),
)
REPLY_MESSAGE = MESSAGE.create_reply(mk.recordset(1, 1, 1))


@dataclass
class IoPairs:
    """The input and output pairs of FleetConnection methods."""

    create_node: tuple[tuple[Any, ...], Any]
    delete_node: tuple[tuple[Any, ...], Any]
    receive: tuple[tuple[Any, ...], Any]
    send: tuple[tuple[Any, ...], Any]
    get_run: tuple[tuple[Any, ...], Any]
    get_fab: tuple[tuple[Any, ...], Any]


# Base TestCase for all connection tests
class FleetConnectionTest(unittest.TestCase):
    """Tests for all connection implementations."""

    # This is to True in each child class
    __test__ = False

    @property
    @abstractmethod
    def send_received(self) -> Message | None:
        """The message received by the server for `send` method."""

    @property
    @abstractmethod
    def get_run_received(self) -> int | None:
        """The run_id received by the server for `get_run` method."""

    @property
    @abstractmethod
    def get_fab_received(self) -> str | None:
        """The fab_hash received by the server for `get_fab` method."""

    @property
    @abstractmethod
    def node_id_received(self) -> int | None:
        """The node_id received by the server for any method."""

    @property
    @abstractmethod
    def connection_type(self) -> type[FleetConnection]:
        """Get the connection type."""

    def setUp(self) -> None:
        """Prepare before each test."""
        # Create a connection
        self.conn = self.connection_type(
            server_address="123.123.123.123:1234",
            insecure=True,
            retry_invoker=RetryInvoker(
                exponential, Exception, max_tries=1, max_time=None
            ),
        )

    def test_create_node(self) -> None:
        """Test create_node method."""
        # Execute
        node_id = self.conn.create_node()

        # Assert
        self.assertEqual(node_id, NODE_ID)

    def test_delete_node(self) -> None:
        """Test delete_node method."""
        # Execute
        self.conn.create_node()
        self.conn.delete_node()

        # Assert
        self.assertEqual(self.node_id_received, NODE_ID)

    def test_receive(self) -> None:
        """Test receive method."""
        # Execute
        self.conn.create_node()
        actual_msg = self.conn.receive()

        # Assert
        assert actual_msg is not None
        self.assertEqual(self.node_id_received, NODE_ID)
        # Message object doesn't support `==` operator
        self.assertEqual(actual_msg.metadata, MESSAGE.metadata)
        self.assertEqual(actual_msg.content, MESSAGE.content)

    def test_send(self) -> None:
        """Test send method."""
        # Execute
        self.conn.create_node()
        self.conn.receive()
        self.conn.send(REPLY_MESSAGE)

        # Assert
        assert self.send_received is not None
        self.assertEqual(self.send_received.metadata, REPLY_MESSAGE.metadata)
        self.assertEqual(self.send_received.content, REPLY_MESSAGE.content)

    def test_get_run(self) -> None:
        """Test get_run method."""
        # Execute
        self.conn.create_node()
        actual_run_info = self.conn.get_run(RUN_ID)

        # Assert
        self.assertEqual(self.get_run_received, RUN_ID)
        self.assertEqual(actual_run_info, RUN_INFO)

    def test_get_fab(self) -> None:
        """Test get_fab method."""
        # Execute
        self.conn.create_node()
        actual_fab = self.conn.get_fab(FAB_HASH, RUN_ID)

        # Assert
        self.assertEqual(self.get_fab_received, FAB_HASH)
        self.assertEqual(actual_fab, FAB)


class GrpcRereFleetConnectionTest(FleetConnectionTest):
    """Tests for GrpcRereFleetConnection."""

    __test__ = True

    @property
    def send_received(self) -> Message | None:
        """The message received by the server for `send` method."""
        return cast(Optional[Message], self._server_received.get("send_received", None))

    @property
    def get_run_received(self) -> int | None:
        """The run_id received by the server for `get_run` method."""
        return cast(Optional[int], self._server_received.get("get_run_received", None))

    @property
    def get_fab_received(self) -> str | None:
        """The fab_hash received by the server for `get_fab` method."""
        return cast(Optional[str], self._server_received.get("get_fab_received", None))

    @property
    def node_id_received(self) -> int | None:
        """The node_id received by the server for any method."""
        return cast(Optional[int], self._server_received.get("node_id_received", None))

    @property
    def connection_type(self) -> type[FleetConnection]:
        """Get the connection type."""
        return GrpcRereFleetConnection

    def setUp(self) -> None:
        """Start to patch the stub."""
        stub = Mock()
        self._server_received: dict[str, Any] = {}

        # Mock RPCs
        stub.Ping.side_effect = self._mock_ping
        stub.CreateNode.side_effect = self._mock_create_node
        stub.DeleteNode.side_effect = self._mock_delete_node
        stub.PullTaskIns.side_effect = self._mock_receive
        stub.PushTaskRes.side_effect = self._mock_send
        stub.GetRun.side_effect = self._mock_get_run
        stub.GetFab.side_effect = self._mock_get_fab

        # Start patcher
        self.patcher = patch(
            "flwr.client.connection.grpc_rere.grpc_rere_fleet_connection.FleetStub",
            return_value=stub,
        )
        self.patcher.start()
        super().setUp()

    def tearDown(self) -> None:
        """Stop the patcher."""
        self.patcher.stop()

    # pylint: disable-next=unused-argument
    def _mock_ping(self, request: PingRequest, **_kwargs: Any) -> PingResponse:
        return PingResponse(success=True)

    def _mock_create_node(
        self,
        request: CreateNodeRequest,  # pylint: disable=unused-argument
        **_kwargs: Any,
    ) -> CreateNodeResponse:
        return CreateNodeResponse(node=Node(node_id=NODE_ID))

    def _mock_delete_node(
        self, request: DeleteNodeRequest, **_kwargs: Any
    ) -> DeleteNodeResponse:
        self._server_received["node_id_received"] = request.node.node_id
        return DeleteNodeResponse()

    def _mock_receive(
        self, request: PullTaskInsRequest, **_kwargs: Any
    ) -> PullTaskInsResponse:
        self._server_received["node_id_received"] = request.node.node_id
        task_ins = serde.message_to_taskins(MESSAGE)
        task_ins.task_id = MESSAGE.metadata.message_id
        return PullTaskInsResponse(task_ins_list=[task_ins])

    def _mock_send(
        self, request: PushTaskResRequest, **_kwargs: Any
    ) -> PushTaskResResponse:
        task_res = request.task_res_list[0]
        msg = serde.message_from_taskres(task_res)
        self._server_received["send_received"] = msg
        return PushTaskResResponse()

    def _mock_get_run(self, request: GetRunRequest, **_kwargs: Any) -> GetRunResponse:
        self._server_received["get_run_received"] = request.run_id
        return GetRunResponse(run=serde.run_to_proto(RUN_INFO))

    def _mock_get_fab(self, request: GetFabRequest, **_kwargs: Any) -> GetFabResponse:
        self._server_received["get_fab_received"] = request.hash_str
        return GetFabResponse(fab=serde.fab_to_proto(FAB))


class GrpcAdapterFleetConnectionTest(GrpcRereFleetConnectionTest):
    """Tests for GrpcAdapterFleetConnection."""

    @property
    def connection_type(self) -> type[FleetConnection]:
        """Get the connection type."""
        return GrpcAdapterFleetConnection

    def setUp(self) -> None:
        """."""
        stub = Mock()
        self._server_received: dict[str, Any] = {}

        def side_effect(request: MessageContainer, **_kwargs: Any) -> MessageContainer:
            req: GrpcMessage | None = None
            res: GrpcMessage | None = None
            # Mock Ping
            if request.grpc_message_name == PingRequest.__qualname__:
                req = PingRequest.FromString(request.grpc_message_content)
                res = self._mock_ping(req)
            # Mock CreateNode
            elif request.grpc_message_name == CreateNodeRequest.__qualname__:
                req = CreateNodeRequest.FromString(request.grpc_message_content)
                res = self._mock_create_node(req)
            # Mock DeleteNode
            elif request.grpc_message_name == DeleteNodeRequest.__qualname__:
                req = DeleteNodeRequest.FromString(request.grpc_message_content)
                res = self._mock_delete_node(req)
            # Mock PullTaskIns (for `receive` method)
            elif request.grpc_message_name == PullTaskInsRequest.__qualname__:
                req = PullTaskInsRequest.FromString(request.grpc_message_content)
                res = self._mock_receive(req)
            # Mock PushTaskRes (for `send` method)
            elif request.grpc_message_name == PushTaskResRequest.__qualname__:
                req = PushTaskResRequest.FromString(request.grpc_message_content)
                res = self._mock_send(req)
            # Mock GetRun
            elif request.grpc_message_name == GetRunRequest.__qualname__:
                req = GetRunRequest.FromString(request.grpc_message_content)
                res = self._mock_get_run(req)
            # Mock GetFab
            elif request.grpc_message_name == GetFabRequest.__qualname__:
                req = GetFabRequest.FromString(request.grpc_message_content)
                res = self._mock_get_fab(req)

            assert res is not None
            return MessageContainer(
                grpc_message_name=res.__class__.__qualname__,
                grpc_message_content=res.SerializeToString(),
            )

        stub.SendReceive.side_effect = side_effect

        # Start patcher
        module = "flwr.client.connection.grpc_adapter.grpc_adapter_fleet_connection"
        self.patcher = patch(
            f"{module}.GrpcAdapterStub",
            return_value=stub,
        )
        self.patcher.start()

        # Create a connection
        self.conn = self.connection_type(
            server_address="123.123.123.123:1234",
            insecure=True,
            retry_invoker=RetryInvoker(
                exponential, Exception, max_tries=1, max_time=None
            ),
        )

    def tearDown(self) -> None:
        """."""
        self.patcher.stop()


class RestFleetConnectionTest(GrpcRereFleetConnectionTest):
    """Tests for RestFleetConnection."""

    @property
    def connection_type(self) -> type[FleetConnection]:
        """Get the connection type."""
        return RestFleetConnection

    def setUp(self) -> None:
        """."""
        self._server_received: dict[str, Any] = {}

        def side_effect(url: str, data: bytes, **_kwargs: Any) -> Mock:
            req: GrpcMessage | None = None
            res: GrpcMessage | None = None
            # Mock Ping
            if url.endswith("ping"):
                req = PingRequest.FromString(data)
                res = self._mock_ping(req)
            # Mock CreateNode
            elif url.endswith("create-node"):
                req = CreateNodeRequest.FromString(data)
                res = self._mock_create_node(req)
            # Mock DeleteNode
            elif url.endswith("delete-node"):
                req = DeleteNodeRequest.FromString(data)
                res = self._mock_delete_node(req)
            # Mock PullTaskIns (for `receive` method)
            elif url.endswith("pull-task-ins"):
                req = PullTaskInsRequest.FromString(data)
                res = self._mock_receive(req)
            # Mock PushTaskRes (for `send` method)
            elif url.endswith("push-task-res"):
                req = PushTaskResRequest.FromString(data)
                res = self._mock_send(req)
            # Mock GetRun
            elif url.endswith("get-run"):
                req = GetRunRequest.FromString(data)
                res = self._mock_get_run(req)
            # Mock GetFab
            elif url.endswith("get-fab"):
                req = GetFabRequest.FromString(data)
                res = self._mock_get_fab(req)

            assert res is not None
            return Mock(
                status_code=200,
                headers={"content-type": "application/protobuf"},
                content=res.SerializeToString(),
            )

        # Start patcher
        self.patcher = patch(
            "requests.post",
        )
        mock_post = self.patcher.start()
        mock_post.side_effect = side_effect

        # Create a connection
        self.conn = self.connection_type(
            server_address="123.123.123.123:1234",
            insecure=True,
            retry_invoker=RetryInvoker(
                exponential, Exception, max_tries=1, max_time=None
            ),
        )

    def tearDown(self) -> None:
        """."""
        self.patcher.stop()
