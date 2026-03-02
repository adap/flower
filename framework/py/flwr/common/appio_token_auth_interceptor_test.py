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
from google.protobuf.message import Message as GrpcMessage

from flwr.common.appio_token_auth_interceptor import (
    AppIoTokenAuthClientInterceptor,
    AppIoTokenAuthServerInterceptor,
    get_authenticated_run_id,
    get_authenticated_token,
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
        invocation_metadata: tuple[tuple[str, str | bytes], ...] = (),
    ) -> None:
        self.method = method
        self.invocation_metadata = invocation_metadata


class _AbortContext:
    """Minimal context object exposing only abort()."""

    def __init__(self) -> None:
        self.abort = Mock(side_effect=grpc.RpcError())


class TestAppIoTokenAuthClientInterceptor(TestCase):
    """Unit tests for the client interceptor."""

    def test_adds_app_token_header(self) -> None:
        """The interceptor adds APP_TOKEN_HEADER metadata."""
        # Reason: enforce the contract that auth token is sent via metadata header.
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

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> str:
            captured["metadata"] = list(client_call_details.metadata or [])
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
        # Reason: avoid breaking unrelated metadata used by other call concerns.
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

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> str:
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
        # Reason: lock the success path for methods that require auth.
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
        # Reason: ensure token-required RPCs fail closed when metadata is absent.
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
        # Reason: guard against unknown tokens even when metadata is present.
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

    def test_verify_token_false_denied(self) -> None:
        """Known token with failed verification yields PERMISSION_DENIED."""
        # Reason: `verify_token` false is a distinct invalid path from missing run_id.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = False
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
        state.get_run_id_by_token.assert_called_once_with("invalid")
        state.verify_token.assert_called_once_with(11, "invalid")

    def test_mismatched_request_run_id_denied(self) -> None:
        """Mismatched request.run_id yields PERMISSION_DENIED."""
        # Reason: ensure token/run binding is enforced as part of auth.
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

    def test_token_exposed_in_authenticated_context(self) -> None:
        """Valid token is available through helper getters inside handler."""
        # Reason: servicers rely on context attributes set by interceptor.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda _request, context: (
                    f"{get_authenticated_run_id(context)}:"
                    f"{get_authenticated_token(context)}"
                )
            ),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(response, "11:valid")

    def test_token_not_required_method_passes_without_metadata(self) -> None:
        """Methods marked as token-optional pass through without metadata."""
        # Reason: superexec RPCs must remain callable without token metadata.
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

    def test_token_not_required_method_ignores_invalid_metadata(self) -> None:
        """Token-optional methods bypass token checks even with bad metadata."""
        # Reason: ensure policy, not metadata presence, controls auth behavior.
        state = Mock()
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": False},
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda _request, _context: "ok"
            ),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "bogus"),),
            ),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(response, "ok")
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_method_not_in_policy_defaults_to_no_token_required(self) -> None:
        """Methods missing in policy map default to pass-through."""
        # Reason: lock current default behavior (`get(method, False)`).
        state = Mock()
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={},
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda _request, _context: "ok"
            ),
            _HandlerCallDetails(method="/flwr.proto.ServerAppIo/GetNodes"),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(response, "ok")
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_non_unary_method_shape_denied_for_token_required_method(self) -> None:
        """Non unary-unary handlers fail closed when token is required."""
        # Reason: interceptor is unary-unary only and must fail closed otherwise.
        state = Mock()
        interceptor = AppIoTokenAuthServerInterceptor(
            state_provider=lambda: state,
            method_requires_token={"/flwr.proto.ServerAppIo/GetNodes": True},
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.stream_stream_rpc_method_handler(
                lambda _request_iter, _context: iter(())
            ),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_token_metadata_bytes_value_is_accepted(self) -> None:
        """Bytes metadata values are decoded and validated."""
        # Reason: cover byte-valued metadata branch in token extraction.
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
                invocation_metadata=((APP_TOKEN_HEADER, b"valid"),),
            ),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=11), Mock())

        self.assertEqual(response, "ok:11")
        state.get_run_id_by_token.assert_called_once_with("valid")
        state.verify_token.assert_called_once_with(11, "valid")

    def test_duplicate_token_headers_use_first_value(self) -> None:
        """First token header value is used when duplicates are present."""
        # Reason: lock parser behavior to avoid silent auth drift.
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
                invocation_metadata=(
                    (APP_TOKEN_HEADER, "first"),
                    (APP_TOKEN_HEADER, "second"),
                ),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )
        state.get_run_id_by_token.assert_called_once_with("first")
        state.verify_token.assert_not_called()

    def test_get_authenticated_run_id_denied_when_context_missing_attr(self) -> None:
        """Missing authenticated run_id in context yields invalid-token error."""
        # Reason: helper must fail closed if interceptor did not set context attrs.
        context = _AbortContext()

        with self.assertRaises(grpc.RpcError):
            get_authenticated_run_id(context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )

    def test_get_authenticated_token_denied_when_context_missing_attr(self) -> None:
        """Missing authenticated token in context yields invalid-token error."""
        # Reason: helper must fail closed if interceptor did not set context attrs.
        context = _AbortContext()

        with self.assertRaises(grpc.RpcError):
            get_authenticated_token(context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, "Invalid token."
        )
