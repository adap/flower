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
from typing import Callable, ContextManager, Optional, Tuple, Union

import grpc

from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.message import Message
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_key_pairs,
)
from flwr.proto import fleet_pb2 as flwr_dot_proto_dot_fleet__pb2
from flwr.proto.fleet_pb2 import CreateNodeRequest

_PUBLIC_KEY_HEADER = "public-key"


class _MockServicer:
    def __init__(self):
        self._lock = threading.Lock()
        self._received_client_metadata = None
        _, self._server_public_key = generate_key_pairs()

    def unary_unary(self, request, context):
        with self._lock:
            self._received_client_metadata = context.invocation_metadata()
            if isinstance(request, CreateNodeRequest):
                context.set_trailing_metadata(
                    ((_PUBLIC_KEY_HEADER, self._server_public_key),)
                )

            return object()

    def received_client_metadata(self):
        with self._lock:
            return self._received_client_metadata


def _generic_handler(servicer: _MockServicer):
    rpc_method_handlers = {
        "CreateNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeRequest.FromString,
            response_serializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeResponse.SerializeToString,
        ),
        "DeleteNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeRequest.FromString,
            response_serializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeResponse.SerializeToString,
        ),
        "PullTaskIns": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsRequest.FromString,
            response_serializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsResponse.SerializeToString,
        ),
        "PushTaskRes": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResRequest.FromString,
            response_serializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResResponse.SerializeToString,
        ),
    }
    return grpc.method_handlers_generic_handler("flwr.proto.Fleet", rpc_method_handlers)


class TestAuthenticateClientInterceptor(unittest.TestCase):
    """Test for client interceptor client authentication."""

    def setUp(self):
        """Initialize mock server and client."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=(("grpc.so_reuseport", int(False)),),
        )
        self._server.add_generic_rpc_handlers((_generic_handler(self._servicer),))
        port = self._server.add_insecure_port("[::]:0")
        self._server.start()

        self._connection: Callable[
            [str, bool, int, Union[bytes, str, None]],
            ContextManager[
                Tuple[
                    Callable[[], Optional[Message]],
                    Callable[[Message], None],
                    Optional[Callable[[], None]],
                    Optional[Callable[[], None]],
                ]
            ],
        ] = grpc_request_response(f"localhost:{port}")
        self._address = f"localhost:{port}"

    def test_client_auth_create_node(self) -> None:
        """Test client authentication during create node."""
        with self._connection(
            self._address,
            True,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
        ) as conn:
            receive, send, create_node, delete_node = conn
            create_node()


if __name__ == "__main__":
    unittest.main(verbosity=2)
