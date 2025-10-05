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
"""Flower server interceptor tests."""


import datetime
import tempfile
import unittest
from typing import Any, Callable

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, now
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    PUBLIC_KEY_HEADER,
    SIGNATURE_HEADER,
    SUPERLINK_NODE_ID,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_HEADER,
    TIMESTAMP_TOLERANCE,
    Status,
)
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_key_pairs,
    public_key_to_bytes,
    sign_message,
)
from flwr.common.typing import RunStatus
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
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
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_res_message
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory

from .node_auth_server_interceptor import NodeAuthServerInterceptor


class TestServerInterceptor(unittest.TestCase):  # pylint: disable=R0902
    """Server interceptor tests."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        self.node_sk, self.node_pk = generate_key_pairs()

        state_factory = LinkStateFactory(":flwr-in-memory-state:")
        self.state = state_factory.state()
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        ffs_factory = FfsFactory(self.tmp_dir.name)
        self.ffs = ffs_factory.ffs()
        objectstore_factory = ObjectStoreFactory()
        self.state.store_node_public_keys({public_key_to_bytes(self.node_pk)})
        self.store = objectstore_factory.store()

        self._server_interceptor = NodeAuthServerInterceptor(state_factory)
        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            ffs_factory,
            objectstore_factory,
            None,
            [self._server_interceptor],
        )

        self._channel = grpc.insecure_channel("localhost:9092")
        self._create_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/CreateNode",
            request_serializer=CreateNodeRequest.SerializeToString,
            response_deserializer=CreateNodeResponse.FromString,
        )
        self._delete_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/DeleteNode",
            request_serializer=DeleteNodeRequest.SerializeToString,
            response_deserializer=DeleteNodeResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullMessages",
            request_serializer=PullMessagesRequest.SerializeToString,
            response_deserializer=PullMessagesResponse.FromString,
        )
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushMessages",
            request_serializer=PushMessagesRequest.SerializeToString,
            response_deserializer=PushMessagesResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._get_run = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetRun",
            request_serializer=GetRunRequest.SerializeToString,
            response_deserializer=GetRunResponse.FromString,
        )
        self._send_node_heartbeat = self._channel.unary_unary(
            "/flwr.proto.Fleet/SendNodeHeartbeat",
            request_serializer=SendNodeHeartbeatRequest.SerializeToString,
            response_deserializer=SendNodeHeartbeatResponse.FromString,
        )
        self._get_fab = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetFab",
            request_serializer=GetFabRequest.SerializeToString,
            response_deserializer=GetFabResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)
        # Cleanup the temp directory
        self.tmp_dir.cleanup()

    def _make_metadata(self) -> list[Any]:
        """Create metadata with signature and timestamp."""
        timestamp = now().isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, public_key_to_bytes(self.node_pk)),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_signature(self) -> list[Any]:
        """Create metadata with invalid signature."""
        timestamp = now().isoformat()
        sk, _ = generate_key_pairs()
        signature = sign_message(sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, public_key_to_bytes(self.node_pk)),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_public_key(self) -> list[Any]:
        """Create metadata with invalid public key."""
        timestamp = now().isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        _, pk = generate_key_pairs()
        return [
            (PUBLIC_KEY_HEADER, public_key_to_bytes(pk)),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_timestamp(self) -> list[Any]:
        """Create metadata with invalid timestamp."""
        timestamp = (
            now()
            - datetime.timedelta(seconds=TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE)
        ).isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, public_key_to_bytes(self.node_pk)),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _test_create_node(self, metadata: list[Any]) -> Any:
        """Test CreateNode."""
        return self._create_node.with_call(
            request=CreateNodeRequest(public_key=public_key_to_bytes(self.node_pk)),
            metadata=metadata,
        )

    def _test_delete_node(self, metadata: list[Any]) -> Any:
        """Test DeleteNode."""
        node_id = self._create_node_in_linkstate()
        req = DeleteNodeRequest(node=Node(node_id=node_id))
        return self._delete_node.with_call(request=req, metadata=metadata)

    def _test_pull_messages(self, metadata: list[Any]) -> Any:
        """Test PullMessages."""
        node_id = self._create_node_in_linkstate()
        req = PullMessagesRequest(node=Node(node_id=node_id))
        return self._pull_messages.with_call(request=req, metadata=metadata)

    def _test_push_messages(self, metadata: list[Any]) -> Any:
        """Test PushMessages."""
        node_id = self._create_node_in_linkstate()
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to running. PushMessages is only allowed in running status.
        self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )
        req = PushMessagesRequest(node=Node(node_id=node_id), messages_list=[msg_proto])
        return self._push_messages.with_call(request=req, metadata=metadata)

    def _test_pull_object(self, metadata: list[Any]) -> Any:
        """Test PullObject."""
        node_id = self._create_node_in_linkstate()
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to running. PushMessages is only allowed in running status.
        self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id="1234"
        )
        return self._pull_object.with_call(request=req, metadata=metadata)

    def _test_push_object(self, metadata: list[Any]) -> Any:
        """Test PushObject."""
        node_id = self._create_node_in_linkstate()
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to running. PushMessages is only allowed in running status.
        self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id="1234",
            object_content=b"1234",
        )
        return self._push_object.with_call(request=req, metadata=metadata)

    def _test_get_run(self, metadata: list[Any]) -> Any:
        """Test GetRun."""
        node_id = self._create_node_in_linkstate()
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to running. GetRun is only allowed in running status.
        self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        req = GetRunRequest(node=Node(node_id=node_id), run_id=run_id)
        return self._get_run.with_call(request=req, metadata=metadata)

    def _test_send_node_heartbeat(self, metadata: list[Any]) -> Any:
        """Test SendNodeHeartbeat."""
        node_id = self._create_node_in_linkstate()
        req = SendNodeHeartbeatRequest(node=Node(node_id=node_id))
        return self._send_node_heartbeat.with_call(request=req, metadata=metadata)

    def _test_get_fab(self, metadata: list[Any]) -> Any:
        """Test GetFab."""
        fab_hash = self.ffs.put(b"mock fab content", {})
        node_id = self._create_node_in_linkstate()
        run_id = self.state.create_run("", "", "", {}, ConfigRecord(), "")
        # Transition status to running. GetFabRequest is only allowed in running status.
        self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        req = GetFabRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            hash_str=fab_hash,
        )
        return self._get_fab.with_call(request=req, metadata=metadata)

    def _create_node_in_linkstate(self) -> int:
        pk_bytes = public_key_to_bytes(self.node_pk)
        return self.state.create_node(public_key=pk_bytes, heartbeat_interval=30)

    @parameterized.expand(
        [
            (_test_create_node,),
            (_test_delete_node,),
            (_test_pull_messages,),
            (_test_push_messages,),
            (_test_pull_object,),
            (_test_push_object,),
            (_test_get_run,),
            (_test_send_node_heartbeat,),
            (_test_get_fab,),
        ]
    )  # type: ignore
    def test_successful_rpc_with_metadata(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC."""
        # Execute
        _, call = rpc(self, self._make_metadata())

        # Assert
        assert call.code() == grpc.StatusCode.OK

    @parameterized.expand(
        [
            (_test_create_node,),
            (_test_delete_node,),
            (_test_pull_messages,),
            (_test_push_messages,),
            (_test_pull_object,),
            (_test_push_object,),
            (_test_get_run,),
            (_test_send_node_heartbeat,),
            (_test_get_fab,),
        ]
    )  # type: ignore
    def test_unsuccessful_rpc_with_invalid_signature(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_signature())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED

    @parameterized.expand(
        [
            (_test_create_node,),
            (_test_delete_node,),
            (_test_pull_messages,),
            (_test_push_messages,),
            (_test_pull_object,),
            (_test_push_object,),
            (_test_get_run,),
            (_test_send_node_heartbeat,),
            (_test_get_fab,),
        ]
    )  # type: ignore
    def test_unsuccessful_rpc_with_invalid_public_key(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_public_key())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED

    @parameterized.expand(
        [
            (_test_create_node,),
            (_test_delete_node,),
            (_test_pull_messages,),
            (_test_push_messages,),
            (_test_pull_object,),
            (_test_push_object,),
            (_test_get_run,),
            (_test_send_node_heartbeat,),
            (_test_get_fab,),
        ]
    )  # type: ignore
    def test_unsuccessful_rpc_with_invalid_timestamp(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_timestamp())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED
