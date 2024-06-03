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
import csv
from logging import WARNING
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import grpc
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_ssh_public_key

from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    generate_shared_key,
    public_key_to_bytes,
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

    def __init__(self, state: State, client_keys_file_path: Path):
        self.state = state
        self.client_keys_file_path = client_keys_file_path

        self.client_public_keys = state.get_client_public_keys()
        if len(self.client_public_keys) == 0:
            log(WARNING, "Authentication enabled, but no known public keys configured")

        private_key = self.state.get_server_private_key()
        public_key = self.state.get_server_public_key()

        if private_key is None or public_key is None:
            raise ValueError("Error loading authentication keys")

        self.server_private_key = bytes_to_private_key(private_key)
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
            self._update_client_keys()
            if client_public_key_bytes not in self.client_public_keys:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            if isinstance(request, CreateNodeRequest):
                return self._create_authenticated_node(
                    client_public_key_bytes, request, context
                )

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
            node_id = self.state.get_node_id(client_public_key_bytes)

            if not self._verify_node_id(node_id, request):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            return method_handler.unary_unary(request, context)  # type: ignore

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )

    def _update_client_keys(self):
        new_known_keys = set()

        try:
            with open(
                self.client_keys_file_path, newline="", encoding="utf-8"
            ) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    for element in row:
                        maybe_public_key = element.encode()
                        public_key = load_ssh_public_key(maybe_public_key)
                        if isinstance(public_key, ec.EllipticCurvePublicKey):
                            new_known_keys.add(public_key_to_bytes(public_key))
                        else:
                            raise ValueError(
                                f"Public key {maybe_public_key} is not an "
                                "elliptic curve public key"
                            )

            self.state.clear_client_public_keys()
            self.state.store_client_public_keys(new_known_keys)
            self.client_public_keys = new_known_keys

        except (ValueError, UnsupportedAlgorithm) as e:
            log(WARNING, f"Abort updating client_public_keys set due to error: {e}")
        except Exception as e:
            log(WARNING, f"Abort updating client_public_keys set due to error: {e}")

    def _verify_node_id(
        self,
        node_id: Optional[int],
        request: Union[
            DeleteNodeRequest,
            PullTaskInsRequest,
            PushTaskResRequest,
            GetRunRequest,
            PingRequest,
        ],
    ) -> bool:
        if node_id is None:
            return False
        if isinstance(request, PushTaskResRequest):
            if len(request.task_res_list) == 0:
                return False
            return request.task_res_list[0].task.producer.node_id == node_id
        if isinstance(request, GetRunRequest):
            return node_id in self.state.get_nodes(request.run_id)
        return request.node.node_id == node_id

    def _verify_hmac(
        self, public_key: ec.EllipticCurvePublicKey, request: Request, hmac_value: bytes
    ) -> bool:
        shared_secret = generate_shared_key(self.server_private_key, public_key)
        return verify_hmac(shared_secret, request.SerializeToString(True), hmac_value)

    def _create_authenticated_node(
        self,
        public_key_bytes: bytes,
        request: CreateNodeRequest,
        context: grpc.ServicerContext,
    ) -> CreateNodeResponse:
        context.send_initial_metadata(
            (
                (
                    _PUBLIC_KEY_HEADER,
                    self.encoded_server_public_key,
                ),
            )
        )

        node_id = self.state.get_node_id(public_key_bytes)

        # Handle `CreateNode` here instead of calling the default method handler
        # Return previously assigned `node_id` for the provided `public_key`
        if node_id is not None:
            self.state.acknowledge_ping(node_id, request.ping_interval)
            return CreateNodeResponse(node=Node(node_id=node_id, anonymous=False))

        # No `node_id` exists for the provided `public_key`
        # Handle `CreateNode` here instead of calling the default method handler
        # Note: the innermost `CreateNode` method will never be called
        node_id = self.state.create_node(request.ping_interval, public_key_bytes)
        return CreateNodeResponse(node=Node(node_id=node_id, anonymous=False))
