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
"""Tests for AppIo token auth interceptors."""

from collections import namedtuple
from unittest import TestCase
from unittest.mock import Mock

import grpc

from flwr.common.appio_token_auth_interceptor import (
    AppIoTokenAuthClientInterceptor,
    AppIoTokenAuthServerInterceptor,
    get_authenticated_run_id,
    verify_authenticated_run_matches_request_run_id,
)
from flwr.common.constant import APP_TOKEN_HEADER
from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


class _HandlerCallDetails:
    def __init__(
        self,
        method: str,
        invocation_metadata: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.method = method
        self.invocation_metadata = invocation_metadata


class TestAppIoTokenAuthClientInterceptor(TestCase):
    """Unit tests for the client interceptor."""

    def test_adds_app_token_header(self) -> None:
        """The interceptor adds APP_TOKEN_HEADER metadata."""
        interceptor = AppIoTokenAuthClientInterceptor(token="abc")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/PullAppInputs",
            timeout=None,
            metadata=None,
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured = {}

        def continuation(client_call_details, request):
            captured["metadata"] = list(client_call_details.metadata or [])
            captured["request"] = request
            return "ok"

        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=details,
            request=GetNodesRequest(run_id=1),
        )

        self.assertEqual(response, "ok")
        self.assertIn((APP_TOKEN_HEADER, "abc"), captured["metadata"])

    def test_preserves_existing_metadata(self) -> None:
        """The interceptor appends metadata and preserves existing entries."""
        interceptor = AppIoTokenAuthClientInterceptor(token="abc")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/PullAppInputs",
            timeout=None,
            metadata=(("x-test", "value"),),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured = {}

        def continuation(client_call_details, request):
            captured["metadata"] = list(client_call_details.metadata or [])
            return "ok"

        interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=details,
            request=GetNodesRequest(run_id=1),
        )

        self.assertIn(("x-test", "value"), captured["metadata"])
        self.assertIn((APP_TOKEN_HEADER, "abc"), captured["metadata"])


class TestAppIoTokenAuthServerInterceptor(TestCase):
    """Unit tests for the server interceptor."""

    def _make_method_handler(self) -> grpc.RpcMethodHandler:
        def handler(request: GetNodesRequest, context: grpc.ServicerContext) -> str:
            verify_authenticated_run_matches_request_run_id(context, request.run_id)
            return f"ok:{get_authenticated_run_id(context)}"

        return grpc.unary_unary_rpc_method_handler(handler)

    def test_valid_token_passes_token_required_method(self) -> None:
        """A valid token allows token-protected methods."""
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )
        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=11), context)

        self.assertEqual(response, "ok:11")
        state.get_run_id_by_token.assert_called_once_with("valid")
        state.verify_token.assert_called_once_with(11, "valid")

    def test_missing_token_denied(self) -> None:
        """Missing token metadata yields PERMISSION_DENIED."""
        state = Mock()
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(method="/flwr.proto.ServerAppIo/GetNodes"),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=11), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )

    def test_invalid_token_denied(self) -> None:
        """Invalid tokens yield PERMISSION_DENIED."""
        state = Mock()
        state.get_run_id_by_token.return_value = None
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "invalid"),),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=11), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )

    def test_mismatched_request_run_id_denied(self) -> None:
        """Mismatched request.run_id yields PERMISSION_DENIED."""
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=99), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )

    def test_token_not_required_method_passes_without_metadata(self) -> None:
        """Methods marked as token-optional pass through without metadata."""
        state = Mock()
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": False},
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda request, context: "ok"
            ),
            _HandlerCallDetails(method="/flwr.proto.ServerAppIo/GetNodes"),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(response, "ok")
        state.get_run_id_by_token.assert_not_called()
