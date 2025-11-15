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
"""Flower Control API license interceptor."""


from collections.abc import Callable, Iterator
from typing import Any

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.license_plugin import LicensePlugin


class ControlLicenseInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Control API interceptor for license checking."""

    def __init__(self, license_plugin: LicensePlugin) -> None:
        """Initialize the interceptor with a license plugin."""
        self.license_plugin = license_plugin

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor license logic.

        Intercept all unary-unary/unary-stream calls from users and check the license.
        Continue RPC call if license check is enabled and passes, else, terminate RPC
        call by setting context to abort.
        """
        # Only apply to Control service
        if not handler_call_details.method.startswith("/flwr.proto.Control/"):
            return continuation(handler_call_details)

        # One of the method handlers in
        # `flwr.superlink.servicer.control.ControlServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._generic_license_unary_method_handler(method_handler)

    def _generic_license_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage | Iterator[GrpcMessage]:
            """Handle the method call with license checking."""
            call = method_handler.unary_unary or method_handler.unary_stream

            if not self.license_plugin.check_license():
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "❗️ License check failed. Please contact the SuperLink "
                    "administrator.",
                )
                raise grpc.RpcError()

            return call(request, context)  # type: ignore

        if method_handler.unary_unary:
            message_handler = grpc.unary_unary_rpc_method_handler
        else:
            message_handler = grpc.unary_stream_rpc_method_handler
        return message_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
