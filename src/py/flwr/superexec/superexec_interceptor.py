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
from typing import Any, Callable, Sequence, Set, Tuple, Union, Type

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.logger import log
from flwr.common.auth_plugin import SuperExecAuthPlugin
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
    LoginRequest,
    LoginResponse,
)

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

Request = Union[
    StartRunRequest,
    StreamLogsRequest,
    LoginRequest,
]

Response = Union[StartRunResponse, StreamLogsResponse, LoginResponse]


class SuperExecInterceptor(grpc.ServerInterceptor):  # type: ignore
    """SuperExec interceptor for user authentication."""

    def __init__(
        self, auth_plugin: SuperExecAuthPlugin,
    ):
        self.auth_plugin = auth_plugin

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
            if not isinstance(request, LoginRequest) and not self.auth_plugin.authenticate(context.invocation_metadata()):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

            return method_handler.unary_unary(request, context)  # type: ignore

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
