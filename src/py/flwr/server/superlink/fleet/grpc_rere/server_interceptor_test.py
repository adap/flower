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
"""Flower server interceptor tests."""


import base64
import unittest

import grpc

from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    GetRunRequest,
    GetRunResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskRes  # pylint: disable=E0611
from flwr.server.app import ADDRESS_FLEET_API_GRPC_RERE, _run_fleet_api_grpc_rere
from flwr.server.superlink.state.state_factory import StateFactory

from .server_interceptor import (
    _AUTH_TOKEN_HEADER,
    _PUBLIC_KEY_HEADER,
    AuthenticateServerInterceptor,
)


class TestServerInterceptor(unittest.TestCase):  # pylint: disable=R0902
    """Server interceptor tests."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        self._client_private_key, self._client_public_key = generate_key_pairs()
        self._server_private_key, self._server_public_key = generate_key_pairs()

        state_factory = StateFactory(":flwr-in-memory-state:")
        self.state = state_factory.state()
        self.state.store_server_private_public_key(
            private_key_to_bytes(self._server_private_key),
            public_key_to_bytes(self._server_public_key),
        )
        self.state.store_client_public_keys(
            {public_key_to_bytes(self._client_public_key)}
        )

        self._server_interceptor = AuthenticateServerInterceptor(self.state)
        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            ADDRESS_FLEET_API_GRPC_RERE, state_factory, None, [self._server_interceptor]
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
        self._pull_task_ins = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullTaskIns",
            request_serializer=PullTaskInsRequest.SerializeToString,
            response_deserializer=PullTaskInsResponse.FromString,
        )
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
        self._ping = self._channel.unary_unary(
            "/flwr.proto.Fleet/Ping",
            request_serializer=PingRequest.SerializeToString,
            response_deserializer=PingResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def test_successful_create_node_with_metadata(self) -> None:
        """Test server interceptor for creating node."""
        # Prepare
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._create_node.with_call(
            request=CreateNodeRequest(),
            metadata=((_PUBLIC_KEY_HEADER, public_key_bytes),),
        )

        expected_metadata = (
            _PUBLIC_KEY_HEADER,
            base64.urlsafe_b64encode(
                public_key_to_bytes(self._server_public_key)
            ).decode(),
        )

        # Assert
        assert call.initial_metadata()[0] == expected_metadata
        assert isinstance(response, CreateNodeResponse)

    def test_unsuccessful_create_node_with_metadata(self) -> None:
        """Test server interceptor for creating node unsuccessfully."""
        # Prepare
        _, client_public_key = generate_key_pairs()
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._create_node.with_call(
                request=CreateNodeRequest(),
                metadata=((_PUBLIC_KEY_HEADER, public_key_bytes),),
            )

    def test_successful_delete_node_with_metadata(self) -> None:
        """Test server interceptor for deleting node."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = DeleteNodeRequest(node=Node(node_id=node_id))
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._delete_node.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        # Assert
        assert isinstance(response, DeleteNodeResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_delete_node_with_metadata(self) -> None:
        """Test server interceptor for deleting node unsuccessfully."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = DeleteNodeRequest(node=Node(node_id=node_id))
        client_private_key, _ = generate_key_pairs()
        shared_secret = generate_shared_key(client_private_key, self._server_public_key)
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._delete_node.with_call(
                request=request,
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )

    def test_successful_pull_task_ins_with_metadata(self) -> None:
        """Test server interceptor for pull task ins."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PullTaskInsRequest(node=Node(node_id=node_id))
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._pull_task_ins.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        # Assert
        assert isinstance(response, PullTaskInsResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_pull_task_ins_with_metadata(self) -> None:
        """Test server interceptor for pull task ins unsuccessfully."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PullTaskInsRequest(node=Node(node_id=node_id))
        client_private_key, _ = generate_key_pairs()
        shared_secret = generate_shared_key(client_private_key, self._server_public_key)
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._pull_task_ins.with_call(
                request=request,
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )

    def test_successful_push_task_res_with_metadata(self) -> None:
        """Test server interceptor for push task res."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PushTaskResRequest(
            task_res_list=[TaskRes(task=Task(producer=Node(node_id=node_id)))]
        )
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._push_task_res.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        # Assert
        assert isinstance(response, PushTaskResResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_push_task_res_with_metadata(self) -> None:
        """Test server interceptor for push task res unsuccessfully."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PushTaskResRequest(
            task_res_list=[TaskRes(task=Task(producer=Node(node_id=node_id)))]
        )
        client_private_key, _ = generate_key_pairs()
        shared_secret = generate_shared_key(client_private_key, self._server_public_key)
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._push_task_res.with_call(
                request=request,
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )

    def test_successful_get_run_with_metadata(self) -> None:
        """Test server interceptor for pull task ins."""
        # Prepare
        self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        run_id = self.state.create_run("", "")
        request = GetRunRequest(run_id=run_id)
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._get_run.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        # Assert
        assert isinstance(response, GetRunResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_get_run_with_metadata(self) -> None:
        """Test server interceptor for pull task ins unsuccessfully."""
        # Prepare
        self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        run_id = self.state.create_run("", "")
        request = GetRunRequest(run_id=run_id)
        client_private_key, _ = generate_key_pairs()
        shared_secret = generate_shared_key(client_private_key, self._server_public_key)
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._get_run.with_call(
                request=request,
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )

    def test_successful_ping_with_metadata(self) -> None:
        """Test server interceptor for pull task ins."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PingRequest(node=Node(node_id=node_id))
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute
        response, call = self._ping.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        # Assert
        assert isinstance(response, PingResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_ping_with_metadata(self) -> None:
        """Test server interceptor for pull task ins unsuccessfully."""
        # Prepare
        node_id = self.state.create_node(
            ping_interval=30, public_key=public_key_to_bytes(self._client_public_key)
        )
        request = PingRequest(node=Node(node_id=node_id))
        client_private_key, _ = generate_key_pairs()
        shared_secret = generate_shared_key(client_private_key, self._server_public_key)
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._ping.with_call(
                request=request,
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )

    def test_successful_restore_node(self) -> None:
        """Test server interceptor for restoring node."""
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )
        response, call = self._create_node.with_call(
            request=CreateNodeRequest(),
            metadata=((_PUBLIC_KEY_HEADER, public_key_bytes),),
        )

        expected_metadata = (
            _PUBLIC_KEY_HEADER,
            base64.urlsafe_b64encode(
                public_key_to_bytes(self._server_public_key)
            ).decode(),
        )

        node = response.node
        client_node_id = node.node_id

        assert call.initial_metadata()[0] == expected_metadata
        assert isinstance(response, CreateNodeResponse)

        request = DeleteNodeRequest(node=node)
        shared_secret = generate_shared_key(
            self._client_private_key, self._server_public_key
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )
        response, call = self._delete_node.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        assert isinstance(response, DeleteNodeResponse)
        assert grpc.StatusCode.OK == call.code()

        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._client_public_key)
        )
        response, call = self._create_node.with_call(
            request=CreateNodeRequest(),
            metadata=((_PUBLIC_KEY_HEADER, public_key_bytes),),
        )

        expected_metadata = (
            _PUBLIC_KEY_HEADER,
            base64.urlsafe_b64encode(
                public_key_to_bytes(self._server_public_key)
            ).decode(),
        )

        assert call.initial_metadata()[0] == expected_metadata
        assert isinstance(response, CreateNodeResponse)
        assert response.node.node_id == client_node_id
