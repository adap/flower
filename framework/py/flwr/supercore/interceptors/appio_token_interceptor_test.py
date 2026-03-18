# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for short-term AppIo token interceptors."""


from collections import namedtuple
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    PushAppMessagesRequest,
    PushAppOutputsRequest,
)
from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611
from flwr.supercore.interceptors import (
    APP_TOKEN_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
    SERVERAPPIO_METHOD_AUTH_POLICY,
    AppIoTokenClientInterceptor,
    AppIoTokenServerInterceptor,
    create_serverappio_token_auth_interceptor,
)

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


class _HandlerCallDetails:
    def __init__(
        self,
        method: str,
        invocation_metadata: tuple[tuple[str, str | bytes], ...],
    ) -> None:
        self.method = method
        self.invocation_metadata = invocation_metadata


class _TokenState:
    def __init__(self, token_to_run_id: dict[str, int]) -> None:
        self._token_to_run_id = token_to_run_id

    def get_run_id_by_token(self, token: str) -> int | None:
        """Return the run id for a token, if present."""
        return self._token_to_run_id.get(token)

    def verify_token(self, run_id: int, token: str) -> bool:
        """Return whether the token is bound to the given run id."""
        return self._token_to_run_id.get(token) == run_id


def _make_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(_request: GrpcMessage, _context: grpc.ServicerContext) -> str:
        return "ok"

    return grpc.unary_unary_rpc_method_handler(_handler)


def _make_non_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(
        _request: GrpcMessage, _context: grpc.ServicerContext
    ) -> list[GrpcMessage]:
        return []

    return grpc.unary_stream_rpc_method_handler(_handler)


class TestAppIoTokenClientInterceptor(TestCase):
    """Unit tests for AppIoTokenClientInterceptor."""

    def test_attach_and_replace_app_token_header(self) -> None:
        """The interceptor should enforce a single App token header."""
        interceptor = AppIoTokenClientInterceptor(token="new-token")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=(("x-test", "value"), (APP_TOKEN_HEADER, "old-token")),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

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
        metadata = captured["metadata"]
        self.assertIn(("x-test", "value"), metadata)
        self.assertEqual(
            [item for item in metadata if item[0] == APP_TOKEN_HEADER],
            [(APP_TOKEN_HEADER, "new-token")],
        )


class TestAppIoTokenServerInterceptor(TestCase):
    """Unit tests for AppIoTokenServerInterceptor."""

    def _new_interceptor(
        self, token_to_run_id: dict[str, int]
    ) -> AppIoTokenServerInterceptor:
        state = _TokenState(token_to_run_id)
        return create_serverappio_token_auth_interceptor(lambda: state)

    @staticmethod
    def _find_serverappio_method(*, requires_token: bool) -> str | None:
        methods = [
            method
            for method, policy in SERVERAPPIO_METHOD_AUTH_POLICY.items()
            if policy.requires_token is requires_token
        ]
        return sorted(methods)[0] if methods else None

    def test_no_auth_method_allows_call_without_token(self) -> None:
        """No-auth methods should pass through without metadata token."""
        interceptor = self._new_interceptor(token_to_run_id={})
        method = self._find_serverappio_method(requires_token=False)
        if method is None:
            self.skipTest("No no-auth ServerAppIo method found in policy table.")

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=(),
            ),
        )

        response = cast(str, intercepted.unary_unary(ListAppsToLaunchRequest(), Mock()))
        self.assertEqual(response, "ok")

    def test_missing_token_denied_for_protected_method(self) -> None:
        """Protected methods should deny requests without token input."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=(),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_invalid_token_denied_for_protected_method(self) -> None:
        """Protected methods should deny requests with invalid tokens."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "invalid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_valid_token_passes_for_protected_method(self) -> None:
        """Protected methods should pass with a valid token."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=1), Mock()))
        self.assertEqual(response, "ok")

    def test_request_token_preferred_over_metadata_token(self) -> None:
        """request.token should be authoritative when non-empty."""
        state = Mock()
        token_to_run_id = {
            "request-token": 9,
            "metadata-token": 1,
        }
        state.get_run_id_by_token.side_effect = token_to_run_id.get

        def _verify_token_side_effect(run_id: int, token: str) -> bool:
            return (token, run_id) == ("request-token", 9)

        def _state_provider() -> Mock:
            return state

        state.verify_token.side_effect = _verify_token_side_effect
        interceptor = AppIoTokenServerInterceptor(
            state_provider=_state_provider,
            method_auth_policy=SERVERAPPIO_METHOD_AUTH_POLICY,
        )

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PushAppOutputs",
                invocation_metadata=((APP_TOKEN_HEADER, "metadata-token"),),
            ),
        )

        response = cast(
            str,
            intercepted.unary_unary(
                PushAppOutputsRequest(token="request-token", run_id=9), Mock()
            ),
        )
        self.assertEqual(response, "ok")
        state.get_run_id_by_token.assert_called_once_with("request-token")

    def test_metadata_fallback_when_request_token_missing(self) -> None:
        """Metadata token should be used if request.token is missing/empty."""
        interceptor = self._new_interceptor(token_to_run_id={"metadata-token": 5})

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PushMessages",
                invocation_metadata=((APP_TOKEN_HEADER, "metadata-token"),),
            ),
        )

        response = cast(
            str,
            intercepted.unary_unary(PushAppMessagesRequest(run_id=5), Mock()),
        )
        self.assertEqual(response, "ok")

    def test_unknown_method_fails_closed(self) -> None:
        """Unknown methods should fail closed with UNAUTHENTICATED."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        continuation = Mock(return_value=_make_unary_handler())
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            continuation,
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/UnknownMethod",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        continuation.assert_not_called()
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_non_unary_handler_fails_closed_for_protected_method(self) -> None:
        """Protected methods with non-unary handlers should fail closed."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_non_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )


class TestMethodPolicyMaps(TestCase):
    """Validate method auth policy map coverage and values."""

    def test_serverappio_policy_has_full_coverage(self) -> None:
        """ServerAppIo policy map should cover all RPC methods exactly."""
        expected_methods = {
            "/flwr.proto.ServerAppIo/ListAppsToLaunch",
            "/flwr.proto.ServerAppIo/RequestToken",
            "/flwr.proto.ServerAppIo/GetRun",
            "/flwr.proto.ServerAppIo/SendAppHeartbeat",
            "/flwr.proto.ServerAppIo/PullAppInputs",
            "/flwr.proto.ServerAppIo/PushAppOutputs",
            "/flwr.proto.ServerAppIo/PushObject",
            "/flwr.proto.ServerAppIo/PullObject",
            "/flwr.proto.ServerAppIo/ConfirmMessageReceived",
            "/flwr.proto.ServerAppIo/UpdateRunStatus",
            "/flwr.proto.ServerAppIo/PushLogs",
            "/flwr.proto.ServerAppIo/PushMessages",
            "/flwr.proto.ServerAppIo/PullMessages",
            "/flwr.proto.ServerAppIo/GetNodes",
        }
        self.assertEqual(set(SERVERAPPIO_METHOD_AUTH_POLICY), expected_methods)

    def test_only_expected_no_auth_methods_exist(self) -> None:
        """Only bootstrap methods should be marked no-auth in the policy table."""
        expected_suffixes = {"ListAppsToLaunch", "RequestToken", "GetRun"}
        no_auth_methods = {
            method.rsplit("/", maxsplit=1)[-1]
            for method, policy in SERVERAPPIO_METHOD_AUTH_POLICY.items()
            if not policy.requires_token
        }
        self.assertEqual(no_auth_methods, expected_suffixes)


class TestFactoryFunctions(TestCase):
    """Validate interceptor factory behavior."""

    def test_serverappio_factory_uses_server_policy(self) -> None:
        """ServerAppIo factory should enforce ServerAppIo policy semantics."""
        state = _TokenState({"valid-token": 1})
        interceptor = create_serverappio_token_auth_interceptor(lambda: state)

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid-token"),),
            ),
        )

        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=1), Mock()))
        self.assertEqual(response, "ok")
