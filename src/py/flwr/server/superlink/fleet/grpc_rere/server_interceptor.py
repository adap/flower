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
"""Flower server interceptor."""


import base64
import threading
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    generate_shared_key,
    verify_hmac,
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
from flwr.server.superlink.state import State

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

Request = Union[
    CreateNodeRequest,
    DeleteNodeRequest,
    PullTaskInsRequest,
    PushTaskResRequest,
    GetRunRequest,
    PingRequest,
]

Response = Union[
    CreateNodeResponse,
    DeleteNodeResponse,
    PullTaskInsResponse,
    PushTaskResResponse,
    GetRunResponse,
    PingResponse,
]


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class AuthenticateServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Server interceptor for client authentication."""

    def __init__(self, state: State):
        self.state = state
        self._lock = threading.Lock()
        self.server_private_key: Optional[ec.EllipticCurvePrivateKey] = None
        private_key = self.state.get_server_private_key()
        public_key = self.state.get_server_public_key()
        if private_key is not None:
            self.server_private_key = bytes_to_private_key(private_key)
        self.client_public_keys = state.get_client_public_keys()
        self.encoded_server_public_key: Optional[bytes] = None
        if public_key is not None:
            self.encoded_server_public_key = base64.urlsafe_b64encode(public_key)

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor authentication logic.

        Intercept all unary calls from clients and authenticate clients by validating
        auth metadata sent by the client. Continue RPC call if client is authenticated,
        else, terminate RPC call by setting context to abort.
        """
        # One of the method handlers in
        # `flwr.server.superlink.fleet.grpc_rere.fleet_server.FleetServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._generic_auth_unary_method_handler(method_handler)

    def _generic_auth_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: Request,
            context: grpc.ServicerContext,
        ) -> Response:
            client_public_key_bytes = base64.urlsafe_b64decode(
                _get_value_from_tuples(
                    _PUBLIC_KEY_HEADER, context.invocation_metadata()
                )
            )
            if client_public_key_bytes not in self.client_public_keys:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            if isinstance(request, CreateNodeRequest):
                return self._create_authenticated_node(
                    client_public_key_bytes, request, context, method_handler
                )

            if isinstance(
                request,
                (
                    DeleteNodeRequest,
                    PullTaskInsRequest,
                    PushTaskResRequest,
                    GetRunRequest,
                    PingRequest,
                ),
            ):
                # Verify hmac value
                hmac_value = base64.urlsafe_b64decode(
                    _get_value_from_tuples(
                        _AUTH_TOKEN_HEADER, context.invocation_metadata()
                    )
                )
                public_key = bytes_to_public_key(client_public_key_bytes)

                if not self._verify_hmac(public_key, request, hmac_value):
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

                # Verify node_id
                try:
                    node_id = self.state.get_node_id(client_public_key_bytes)
                except KeyError:
                    node_id = -1

                if not self._verify_node_id(node_id, request):
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            else:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            return method_handler.unary_unary(request, context)  # type: ignore

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )

    def _verify_node_id(self, node_id: int, request: Request) -> bool:
        if isinstance(request, CreateNodeRequest):
            return False
        if isinstance(request, PushTaskResRequest):
            return request.task_res_list[0].task.consumer.node_id == node_id
        if isinstance(request, GetRunRequest):
            return node_id in self.state.get_nodes(request.run_id)
        return request.node.node_id == node_id

    def _verify_hmac(
        self, public_key: ec.EllipticCurvePublicKey, request: Request, hmac_value: bytes
    ) -> bool:
        if self.server_private_key is None:
            return False

        shared_secret = generate_shared_key(self.server_private_key, public_key)
        return verify_hmac(shared_secret, request.SerializeToString(True), hmac_value)

    def _create_authenticated_node(
        self,
        public_key_bytes: bytes,
        request: CreateNodeRequest,
        context: grpc.ServicerContext,
        method_handler: grpc.RpcMethodHandler,
    ) -> CreateNodeResponse:
        context.send_initial_metadata(
            (
                (
                    _PUBLIC_KEY_HEADER,
                    self.encoded_server_public_key,
                ),
            )
        )
        try:
            node_id = self.state.get_node_id(public_key_bytes)
        except KeyError:
            node_id = None

        if node_id is not None:
            with self._lock:
                # Handle RPC here instead of passing to default method_handler
                self.state.restore_node(node_id, request.ping_interval)
                return CreateNodeResponse(node=Node(node_id=node_id, anonymous=False))

        response: CreateNodeResponse = method_handler.unary_unary(request, context)

        with self._lock:
            self.state.store_node_id_and_public_key(response.node.node_id, public_key_bytes)
            return response
