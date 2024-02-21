# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower client interceptor tests."""

import threading
import unittest
from concurrent import futures
from typing import Optional, Sequence, Tuple, Union

import grpc

from flwr.client.grpc_rere_client.connection import init_grpc_request_response
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    public_key_to_bytes,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)

from .client_interceptor import (
    _AUTH_TOKEN_HEADER,
    _PUBLIC_KEY_HEADER,
    AuthenticateClientInterceptor,
    Request,
)


class _MockServicer:
    """Mock Servicer for Flower clients."""

    def __init__(self) -> None:
        """Initialize mock servicer."""
        self._lock = threading.Lock()
        self._received_client_metadata: Optional[
            Sequence[Tuple[str, Union[str, bytes]]]
        ] = None
        self.server_private_key, self.server_public_key = generate_key_pairs()
        self._received_message_bytes: bytes = b""

    def unary_unary(self, request: Request, context: grpc.ServicerContext) -> object:
        """Handle unary call."""
        with self._lock:
            self._received_client_metadata = context.invocation_metadata()
            self._received_message_bytes = request.SerializeToString(True)
            if isinstance(request, CreateNodeRequest):
                context.set_trailing_metadata(
                    ((_PUBLIC_KEY_HEADER, self.server_public_key),)
                )

            return object()

    def received_client_metadata(
        self,
    ) -> Optional[Sequence[Tuple[str, Union[str, bytes]]]]:
        """Return received client metadata."""
        with self._lock:
            return self._received_client_metadata

    def received_message_bytes(self) -> bytes:
        """Return received message bytes."""
        with self._lock:
            return self._received_message_bytes


def _add_generic_handler(servicer: _MockServicer, server: grpc.Server) -> None:
    rpc_method_handlers = {
        "CreateNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=CreateNodeRequest.FromString,
            response_serializer=CreateNodeResponse.SerializeToString,
        ),
        "DeleteNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=DeleteNodeRequest.FromString,
            response_serializer=DeleteNodeResponse.SerializeToString,
        ),
        "PullTaskIns": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PullTaskInsRequest.FromString,
            response_serializer=PullTaskInsResponse.SerializeToString,
        ),
        "PushTaskRes": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PushTaskResRequest.FromString,
            response_serializer=PushTaskResResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "flwr.proto.Fleet", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


class TestAuthenticateClientInterceptor(unittest.TestCase):
    """Test for client interceptor client authentication."""

    def setUp(self) -> None:
        """Initialize mock server and client."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=(("grpc.so_reuseport", int(False)),),
        )
        self._servicer = _MockServicer()
        _add_generic_handler(self._servicer, self._server)
        port = self._server.add_insecure_port("[::]:0")
        self._server.start()
        self._client_private_key, self._client_public_key = generate_key_pairs()
        self._client_interceptor = AuthenticateClientInterceptor(
            self._client_private_key, self._client_public_key
        )

        self._connection = init_grpc_request_response(self._client_interceptor)
        self._address = f"localhost:{port}"

    def test_client_auth_create_node(self) -> None:
        """Test client authentication during create node."""
        with self._connection(
            self._address,
            True,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
        ) as conn:
            _, _, create_node, _ = conn
            assert create_node is not None
            create_node()
            expected_client_metadata = (
                _PUBLIC_KEY_HEADER,
                public_key_to_bytes(self._client_public_key),
            )
            assert self._servicer.received_client_metadata() == expected_client_metadata
            assert (
                self._client_interceptor.server_public_key
                == self._servicer.server_public_key
            )

    def test_client_auth_delete_node(self) -> None:
        """Test client authentication during delete node."""
        with self._connection(
            self._address,
            True,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
        ) as conn:
            _, _, _, delete_node = conn
            assert delete_node is not None
            delete_node()
            shared_secret = generate_shared_key(
                self._servicer.server_private_key, self._client_public_key
            )
            expected_hmac = compute_hmac(
                shared_secret, self._servicer.received_message_bytes()
            )
            expected_client_metadata = (
                _AUTH_TOKEN_HEADER,
                expected_hmac,
            )
            assert self._servicer.received_client_metadata() == expected_client_metadata


if __name__ == "__main__":
    unittest.main(verbosity=2)
