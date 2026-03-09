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
"""Tests for AppIo auth interceptors."""

from collections import namedtuple
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import APP_TOKEN_HEADER
from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611
from flwr.supercore.auth.appio_auth import AuthInput, CallerIdentity, TokenAuthenticator
from flwr.supercore.auth.constant import (
    APPIO_SIGNED_METADATA_METHOD_HEADER,
    APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER,
    APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER,
    APPIO_SIGNED_METADATA_SIGNATURE_HEADER,
    APPIO_SIGNED_METADATA_TIMESTAMP_HEADER,
    AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA,
    AUTH_MECHANISM_TOKEN,
    AUTHENTICATION_FAILED_MESSAGE,
    CALLER_TYPE_APP_EXECUTOR,
    CALLER_TYPE_SUPEREXEC,
)
from flwr.supercore.auth.policy import MethodAuthPolicy
from flwr.supercore.interceptors.appio_auth_interceptor import (
    AppIoAuthClientInterceptor,
    AppIoAuthServerInterceptor,
    get_authenticated_caller_identity,
    get_authenticated_run_id,
    get_authenticated_token,
    verify_authenticated_run_matches_request_run_id,
)

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


class _SignedMetadataProbeAuthenticator:
    """Test authenticator to validate signed-metadata extraction behavior."""

    mechanism = AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA

    def __init__(self) -> None:
        self.last_auth_input: AuthInput | None = None

    def is_present(self, auth_input: AuthInput) -> bool:
        """Return whether signed metadata auth input was supplied."""
        self.last_auth_input = auth_input
        return auth_input.signed_metadata_present

    def authenticate(self, auth_input: AuthInput) -> CallerIdentity | None:
        """Return synthetic caller identity when signed metadata is complete."""
        if auth_input.signed_metadata is None:
            return None
        return CallerIdentity(
            mechanism=self.mechanism,
            caller_type=CALLER_TYPE_SUPEREXEC,
            key_fingerprint="probe",
        )


class TestAppIoAuthClientInterceptor(TestCase):
    """Unit tests for the client interceptor."""

    def test_adds_app_token_header(self) -> None:
        """The interceptor adds APP_TOKEN_HEADER metadata."""
        # Reason: enforce the contract that auth token is sent via metadata header.
        interceptor = AppIoAuthClientInterceptor(token="abc")
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
        interceptor = AppIoAuthClientInterceptor(token="abc")
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


class TestAppIoAuthServerInterceptor(TestCase):
    """Unit tests for the server interceptor."""

    def _make_token_interceptor(
        self,
        state: Mock,
        method_auth_policy: dict[str, MethodAuthPolicy],
    ) -> AppIoAuthServerInterceptor:
        return AppIoAuthServerInterceptor(
            method_auth_policy=method_auth_policy,
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
            },
        )

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
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_invalid_token_denied(self) -> None:
        """Invalid tokens yield PERMISSION_DENIED."""
        # Reason: guard against unknown tokens even when metadata is present.
        state = Mock()
        state.get_run_id_by_token.return_value = None
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_verify_token_false_denied(self) -> None:
        """Known token with failed verification yields PERMISSION_DENIED."""
        # Reason: `verify_token` false is a distinct invalid path from missing run_id.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = False
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        state.get_run_id_by_token.assert_called_once_with("invalid")
        state.verify_token.assert_called_once_with(11, "invalid")

    def test_mismatched_request_run_id_denied(self) -> None:
        """Mismatched request.run_id yields PERMISSION_DENIED."""
        # Reason: ensure token/run binding is enforced as part of auth.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_token_exposed_in_authenticated_context(self) -> None:
        """Valid token is available through helper getters inside handler."""
        # Reason: servicers rely on context attributes set by interceptor.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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

    def test_caller_identity_exposed_in_authenticated_context(self) -> None:
        """Valid token stores normalized caller identity in context."""
        # Reason: non-token mechanisms will also consume caller identity helpers.
        state = Mock()
        state.get_run_id_by_token.return_value = 11
        state.verify_token.return_value = True
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda _request, context: (
                    f"{get_authenticated_caller_identity(context).caller_type}:"
                    f"{get_authenticated_caller_identity(context).mechanism}:"
                    f"{get_authenticated_caller_identity(context).run_id}"
                )
            ),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(
            response, f"{CALLER_TYPE_APP_EXECUTOR}:{AUTH_MECHANISM_TOKEN}:11"
        )

    def test_token_not_required_method_passes_without_metadata(self) -> None:
        """Methods marked as token-optional pass through without metadata."""
        # Reason: superexec RPCs must remain callable without token metadata.
        state = Mock()
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.no_auth()
            },
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
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.no_auth()
            },
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

    def test_method_not_in_policy_is_denied(self) -> None:
        """Methods missing in policy map are denied."""
        # Reason: fail closed so newly added RPCs cannot bypass auth by omission.
        state = Mock()
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={},
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(method="/flwr.proto.ServerAppIo/GetNodes"),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_interceptor_constructor_fails_on_policy_authenticator_mismatch(
        self,
    ) -> None:
        """Policy/authenticator mismatches fail fast at interceptor construction."""
        # Reason: fail during startup instead of silently dropping auth mechanisms.
        with self.assertRaisesRegex(
            ValueError, "references mechanisms without authenticators"
        ):
            AppIoAuthServerInterceptor(
                method_auth_policy={
                    "/flwr.proto.ServerAppIo/GetNodes": (
                        MethodAuthPolicy.token_required()
                    )
                },
                authenticators={},
            )

    def test_non_unary_method_shape_denied_for_token_required_method(self) -> None:
        """Non unary-unary handlers fail closed when token is required."""
        # Reason: interceptor is unary-unary only and must fail closed otherwise.
        state = Mock()
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_token_metadata_bytes_value_is_denied(self) -> None:
        """Bytes metadata values are rejected for string-token-only auth."""
        # Reason: lock current string-only metadata contract for token handling.
        state = Mock()
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, b"valid"),),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=11), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        state.get_run_id_by_token.assert_not_called()
        state.verify_token.assert_not_called()

    def test_duplicate_token_headers_use_last_value(self) -> None:
        """Last token header value is used when duplicates are present."""
        # Reason: lock parser behavior to avoid silent auth drift.
        state = Mock()
        state.get_run_id_by_token.return_value = None
        interceptor = self._make_token_interceptor(
            state=state,
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": MethodAuthPolicy.token_required()
            },
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
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        state.get_run_id_by_token.assert_called_once_with("second")
        state.verify_token.assert_not_called()

    def test_signed_metadata_absent_is_marked_not_present(self) -> None:
        """No signed-metadata headers should be tracked as absent."""
        # Reason: auth engine must distinguish truly absent input from malformed input.
        state = Mock()
        probe_authenticator = _SignedMetadataProbeAuthenticator()
        interceptor = AppIoAuthServerInterceptor(
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": (
                    MethodAuthPolicy.signed_metadata_required()
                )
            },
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: probe_authenticator,
            },
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(method="/flwr.proto.ServerAppIo/GetNodes"),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        self.assertIsNotNone(probe_authenticator.last_auth_input)
        auth_input = probe_authenticator.last_auth_input
        assert auth_input is not None
        self.assertFalse(auth_input.signed_metadata_present)
        self.assertIsNone(auth_input.signed_metadata)

    def test_signed_metadata_partial_headers_are_marked_present_but_malformed(
        self,
    ) -> None:
        """Partial signed metadata should be marked present and denied."""
        # Reason: malformed input should not collapse into "missing input" semantics.
        state = Mock()
        probe_authenticator = _SignedMetadataProbeAuthenticator()
        interceptor = AppIoAuthServerInterceptor(
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": (
                    MethodAuthPolicy.signed_metadata_required()
                )
            },
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: probe_authenticator,
            },
        )

        intercepted = interceptor.intercept_service(
            lambda _: self._make_method_handler(),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER, b"pk"),),
            ),
        )
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
        self.assertIsNotNone(probe_authenticator.last_auth_input)
        auth_input = probe_authenticator.last_auth_input
        assert auth_input is not None
        self.assertTrue(auth_input.signed_metadata_present)
        self.assertIsNone(auth_input.signed_metadata)

    def test_signed_metadata_complete_headers_build_signed_metadata_input(self) -> None:
        """Complete signed metadata should be parsed into AuthInput.signed_metadata."""
        # Reason: keep extraction logic local in interceptor for future
        # mechanism support.
        state = Mock()
        probe_authenticator = _SignedMetadataProbeAuthenticator()
        interceptor = AppIoAuthServerInterceptor(
            method_auth_policy={
                "/flwr.proto.ServerAppIo/GetNodes": (
                    MethodAuthPolicy.signed_metadata_required()
                )
            },
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: probe_authenticator,
            },
        )

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(
                lambda _request, _context: "ok"
            ),
            _HandlerCallDetails(
                method="/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=(
                    (APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER, b"pk"),
                    (APPIO_SIGNED_METADATA_SIGNATURE_HEADER, b"sig"),
                    (APPIO_SIGNED_METADATA_TIMESTAMP_HEADER, "2026-03-09T10:00:00"),
                    (
                        APPIO_SIGNED_METADATA_METHOD_HEADER,
                        "/flwr.proto.ServerAppIo/GetNodes",
                    ),
                    (APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER, "serverapp"),
                ),
            ),
        )

        response = intercepted.unary_unary(GetNodesRequest(run_id=1), Mock())

        self.assertEqual(response, "ok")
        self.assertIsNotNone(probe_authenticator.last_auth_input)
        auth_input = probe_authenticator.last_auth_input
        assert auth_input is not None
        self.assertTrue(auth_input.signed_metadata_present)
        self.assertIsNotNone(auth_input.signed_metadata)
        signed_metadata = auth_input.signed_metadata
        assert signed_metadata is not None
        self.assertEqual(signed_metadata.public_key, b"pk")
        self.assertEqual(signed_metadata.signature, b"sig")

    def test_get_authenticated_run_id_denied_when_context_missing_attr(self) -> None:
        """Missing authenticated run_id in context yields invalid-token error."""
        # Reason: helper must fail closed if interceptor did not set context attrs.
        context = _AbortContext()

        with self.assertRaises(grpc.RpcError):
            get_authenticated_run_id(context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_get_authenticated_caller_identity_denied_when_context_missing(
        self,
    ) -> None:
        """Missing caller identity in context yields invalid-token error."""
        # Reason: caller identity helper must also fail closed.
        context = _AbortContext()

        with self.assertRaises(grpc.RpcError):
            get_authenticated_caller_identity(context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_get_authenticated_token_denied_when_context_missing_attr(self) -> None:
        """Missing authenticated token in context yields invalid-token error."""
        # Reason: helper must fail closed if interceptor did not set context attrs.
        context = _AbortContext()

        with self.assertRaises(grpc.RpcError):
            get_authenticated_token(context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE
        )
