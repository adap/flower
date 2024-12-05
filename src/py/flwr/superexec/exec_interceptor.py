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
"""Flower Exec API interceptor."""


from typing import Any, Callable, Union

import grpc

from flwr.common.auth_plugin import ExecAuthPlugin
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokenRequest,
    GetAuthTokenResponse,
    LoginRequest,
    LoginResponse,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)

_AUTH_TOKEN_HEADER = "access-token"
_REFRESH_TOKEN_HEADER = "refresh-token"

Request = Union[
    StartRunRequest,
    StreamLogsRequest,
    LoginRequest,
    GetAuthTokenRequest,
]

Response = Union[
    StartRunResponse, StreamLogsResponse, LoginResponse, GetAuthTokenResponse
]


class SuperExecInterceptor(grpc.ServerInterceptor):  # type: ignore
    """SuperExec interceptor for user authentication."""

    def __init__(
        self,
        auth_plugin: ExecAuthPlugin,
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
            if isinstance(
                request, (LoginRequest, GetAuthTokenRequest)
            ) or self.auth_plugin.authenticate(context.invocation_metadata()):
                return method_handler.unary_unary(request, context)  # type: ignore

            if self.auth_plugin.refresh_token(context):
                print(context)
                return method_handler.unary_unary(request, context)  # type: ignore

            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
