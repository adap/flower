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
from logging import WARNING
from typing import Any, Callable, Sequence, Tuple, Union

import grpc

from flwr.common.logger import log
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
            is_public_key_known = client_public_key_bytes in self.client_public_keys
            if not is_public_key_known:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            if isinstance(request, CreateNodeRequest):
                context.send_initial_metadata(
                    (
                        (
                            _PUBLIC_KEY_HEADER,
                            self.encoded_server_public_key,
                        ),
                    )
                )
            elif isinstance(
                request,
                (
                    DeleteNodeRequest,
                    PullTaskInsRequest,
                    PushTaskResRequest,
                    GetRunRequest,
                    PingRequest,
                ),
            ):
                hmac_value = base64.urlsafe_b64decode(
                    _get_value_from_tuples(
                        _AUTH_TOKEN_HEADER, context.invocation_metadata()
                    )
                )
                client_public_key = bytes_to_public_key(client_public_key_bytes)
                shared_secret = generate_shared_key(
                    self.server_private_key,
                    client_public_key,
                )
                verify = verify_hmac(
                    shared_secret, request.SerializeToString(True), hmac_value
                )
                if not verify:
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")
            else:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            return method_handler.unary_unary(request, context)  # type: ignore

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
