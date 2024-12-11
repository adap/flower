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
"""Flower run interceptor."""


from typing import Any, Callable, Union

import grpc

from flwr.common.auth_plugin import CliAuthPlugin
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StreamLogsRequest,
)

Request = Union[
    StartRunRequest,
    StreamLogsRequest,
]


class CliUserAuthInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor  # type: ignore
):
    """CLI interceptor for user authentication."""

    def __init__(self, auth_plugin: CliAuthPlugin):
        self.auth_plugin = auth_plugin

    def _authenticated_call(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Send and receive tokens via metadata."""
        new_metadata = self.auth_plugin.write_tokens_to_metadata(
            client_call_details.metadata or []
        )

        details = client_call_details._replace(metadata=new_metadata)

        response = continuation(details, request)
        if response.initial_metadata():
            retrieved_metadata = dict(response.initial_metadata())
            self.auth_plugin.store_tokens(retrieved_metadata)

        return response

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Intercept a unary-unary call for user authentication.

        This method intercepts a unary-unary RPC call initiated from the CLI and adds
        the required authentication tokens to the RPC metadata.
        """
        return self._authenticated_call(continuation, client_call_details, request)

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Intercept a unary-stream call for user authentication.

        This method intercepts a unary-stream RPC call initiated from the CLI and adds
        the required authentication tokens to the RPC metadata.
        """
        return self._authenticated_call(continuation, client_call_details, request)
