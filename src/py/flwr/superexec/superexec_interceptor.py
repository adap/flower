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
"""Flower superexec interceptor."""


import base64
from logging import WARNING
from typing import Any, Callable, Optional, Sequence, Set, Tuple, Union

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    generate_shared_key,
    verify_hmac,
)
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

Request = Union[
    StartRunRequest,
    StreamLogsRequest,
]

Response = Union[StartRunResponse, StreamLogsResponse]


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class SuperExecInterceptor(grpc.ServerInterceptor):  # type: ignore
    """SuperExec interceptor for user authentication."""

    def __init__(
        self, user_public_keys: Set[bytes], private_key: bytes, public_key: bytes
    ):
        self.user_public_keys = user_public_keys
        if len(self.user_public_keys) == 0:
            log(WARNING, "Authentication enabled, but no known public keys configured")

        self.server_private_key = bytes_to_private_key(private_key)
        self.encoded_server_public_key = base64.urlsafe_b64encode(public_key)

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor authentication logic.

        Intercept all unary calls from users and authenticate users by validating auth
        metadata sent by the user. Continue RPC call if user is authenticated, else,
        terminate RPC call by setting context to abort.
        """
        # One of the method handlers in
        # `flwr.superexec.exec_servicer.ExecServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._generic_auth_unary_method_handler(method_handler)

    def _generic_auth_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: Request,
            context: grpc.ServicerContext,
        ) -> Response:
            user_public_key_bytes = base64.urlsafe_b64decode(
                _get_value_from_tuples(
                    _PUBLIC_KEY_HEADER, context.invocation_metadata()
                )
            )
            if user_public_key_bytes not in self.user_public_keys:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            # Verify hmac value
            hmac_value = base64.urlsafe_b64decode(
                _get_value_from_tuples(
                    _AUTH_TOKEN_HEADER, context.invocation_metadata()
                )
            )
            public_key = bytes_to_public_key(user_public_key_bytes)

            if not self._verify_hmac(public_key, request, hmac_value):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            return method_handler.unary_unary(request, context)  # type: ignore

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )

    def _verify_hmac(
        self, public_key: ec.EllipticCurvePublicKey, request: Request, hmac_value: bytes
    ) -> bool:
        shared_secret = generate_shared_key(self.server_private_key, public_key)
        return verify_hmac(shared_secret, request.SerializeToString(True), hmac_value)
