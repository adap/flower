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
"""General AppIo authentication interceptors shared by AppIo services.

Design note:
- Keep one enforcement point for AppIo auth decisions in this interceptor.
- Avoid stacking mechanism-specific interceptors, which can introduce ordering
  and interaction bugs as mechanisms are added.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import APP_TOKEN_HEADER
from flwr.supercore.auth.appio_auth import (
    AuthDecisionEngine,
    Authenticator,
    AuthInput,
    CallerIdentity,
    SignedMetadataAuthInput,
)
from flwr.supercore.auth.constant import (
    APPIO_SIGNED_METADATA_METHOD_HEADER,
    APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER,
    APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER,
    APPIO_SIGNED_METADATA_SIGNATURE_HEADER,
    APPIO_SIGNED_METADATA_TIMESTAMP_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
)
from flwr.supercore.auth.policy import MethodAuthPolicy

# Keep a single canonical auth failure message.
# Do not vary details by failure mode (missing token, unknown token, run mismatch,
# malformed metadata, etc.), or callers could use error differences as an auth
# oracle.
_AUTH_CALLER_IDENTITY_CTX_ATTR = "_flwr_appio_authenticated_caller_identity"
_AUTH_RUN_ID_CTX_ATTR = "_flwr_appio_authenticated_run_id"
_AUTH_TOKEN_CTX_ATTR = "_flwr_appio_authenticated_token"


def _abort_auth_denied(context: grpc.ServicerContext) -> None:
    # Use this only when we already have a live ServicerContext inside an
    # executing RPC. This complements `_permission_denied_terminator`, which is
    # used while building the handler in `intercept_service`.
    """Abort current RPC with canonical AppIo auth denied status/details."""
    context.abort(grpc.StatusCode.PERMISSION_DENIED, AUTHENTICATION_FAILED_MESSAGE)
    raise grpc.RpcError()


def _permission_denied_terminator(message: str) -> grpc.RpcMethodHandler:
    # `intercept_service` must return an RpcMethodHandler. When auth fails before
    # we can safely invoke the real handler, return a tiny handler that aborts
    # with PERMISSION_DENIED at call execution time.
    """Return a unary-unary handler that immediately aborts the RPC."""

    def terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, message)
        raise grpc.RpcError()

    return grpc.unary_unary_rpc_method_handler(terminate)


def _extract_token_from_metadata(
    metadata: Sequence[tuple[str, str | bytes]] | None,
) -> str | None:
    # Metadata parsing is intentionally conservative: if the expected header is
    # missing/unusable, treat it as invalid token. Do not expose a distinct
    # "malformed token metadata" error (oracle risk).
    # The current token contract is string-only for APP_TOKEN_HEADER.
    # Byte-valued metadata is treated as invalid input.
    """Read App token from invocation metadata."""
    token = dict(metadata or ()).get(APP_TOKEN_HEADER)
    return token if isinstance(token, str) else None


def _extract_signed_metadata_from_metadata(
    metadata: Sequence[tuple[str, str | bytes]] | None,
) -> tuple[bool, SignedMetadataAuthInput | None]:
    """Extract signed metadata input and explicit presence state."""
    md_map = dict(metadata or ())
    public_key = md_map.get(APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER)
    signature = md_map.get(APPIO_SIGNED_METADATA_SIGNATURE_HEADER)
    timestamp = md_map.get(APPIO_SIGNED_METADATA_TIMESTAMP_HEADER)
    method = md_map.get(APPIO_SIGNED_METADATA_METHOD_HEADER)
    plugin_type = md_map.get(APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER)

    required_values = (public_key, signature, timestamp, method)
    if all(value is None for value in required_values):
        return False, None
    if any(value is None for value in required_values):
        # Preserve the distinction between "absent" and "present but malformed".
        return True, None
    if (
        not isinstance(public_key, bytes)
        or not isinstance(signature, bytes)
        or not isinstance(timestamp, str)
        or not isinstance(method, str)
    ):
        return True, None
    if plugin_type is not None and not isinstance(plugin_type, str):
        return True, None

    return (
        True,
        SignedMetadataAuthInput(
            public_key=public_key,
            signature=signature,
            timestamp_iso=timestamp,
            method=method,
            plugin_type=plugin_type,
        ),
    )


def get_authenticated_caller_identity(context: grpc.ServicerContext) -> CallerIdentity:
    """Get caller identity set by AppIo auth interceptor."""
    caller_identity = getattr(context, _AUTH_CALLER_IDENTITY_CTX_ATTR, None)
    if caller_identity is None:
        _abort_auth_denied(context)
    return cast(CallerIdentity, caller_identity)


def get_authenticated_run_id(context: grpc.ServicerContext) -> int:
    """Get authenticated run_id from AppIo auth interceptor context."""
    run_id = get_authenticated_caller_identity(context).run_id
    if run_id is None:
        _abort_auth_denied(context)
    return cast(int, run_id)


def get_authenticated_token(context: grpc.ServicerContext) -> str:
    """Get token set by AppIo auth interceptor.

    Token-only helper: this aborts for non-token authenticated callers and
    should not be used in mechanism-agnostic code paths.
    """
    token = getattr(context, _AUTH_TOKEN_CTX_ATTR, None)
    if token is None:
        _abort_auth_denied(context)
    return cast(str, token)


def verify_authenticated_run_matches_request_run_id(
    context: grpc.ServicerContext, request_run_id: int
) -> int:
    # Treat run/token binding mismatches as generic invalid-token auth failures.
    # Do not return "wrong run_id" details, which would leak run
    # existence/binding info.
    """Verify request.run_id matches interceptor-authenticated run_id."""
    authenticated_run_id = get_authenticated_run_id(context)
    if authenticated_run_id != request_run_id:
        _abort_auth_denied(context)
    return authenticated_run_id


class AppIoAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach AppIo authentication metadata to outbound unary RPCs.

    Current implementation injects token metadata. Keeping this class general avoids
    introducing one client interceptor per mechanism.

    This interceptor is responsible for attaching auth metadata/material to outgoing
    RPCs.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Attach App token in metadata and continue RPC."""
        metadata = list(client_call_details.metadata or [])
        # Remove any existing App token headers to avoid duplicates and ensure
        # this interceptor's token is the single authoritative value.
        metadata = [(key, value) for key, value in metadata if key != APP_TOKEN_HEADER]
        metadata.append((APP_TOKEN_HEADER, self._token))
        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)


class AppIoAuthServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Validate AppIo auth metadata based on per-method policy.

    This interceptor is the single AppIo auth enforcement point. It delegates mechanism
    checks to the decision engine/authenticators, then attaches a normalized caller
    identity for downstream servicer logic.

    This interceptor is responsible for extracting AuthInputs from incoming RPCs.
    """

    def __init__(
        self,
        method_auth_policy: Mapping[str, MethodAuthPolicy],
        authenticators: Mapping[str, Authenticator],
    ) -> None:
        self._auth_decision_engine = AuthDecisionEngine(
            authenticators=authenticators,
            method_auth_policy=method_auth_policy,
        )
        self._method_auth_policy = method_auth_policy

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Validate metadata for methods configured as auth-protected."""
        method = handler_call_details.method
        method_policy = self._method_auth_policy.get(method)
        if method_policy is None:
            # Fail closed for unknown methods: policy tables must explicitly
            # classify every exposed RPC to avoid accidental unauthenticated
            # access when methods are added.
            return _permission_denied_terminator(AUTHENTICATION_FAILED_MESSAGE)
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)

        if not method_policy.requires_authentication:
            return method_handler
        if method_handler.unary_unary is None:
            # This interceptor currently protects unary-unary AppIo RPCs only.
            # If method shape is unexpected, fail closed with the same auth
            # error.
            return _permission_denied_terminator(AUTHENTICATION_FAILED_MESSAGE)
        unary_unary_handler = cast(
            Callable[[GrpcMessage, grpc.ServicerContext], GrpcMessage],
            method_handler.unary_unary,
        )

        token = _extract_token_from_metadata(handler_call_details.invocation_metadata)
        signed_metadata_present, signed_metadata = (
            _extract_signed_metadata_from_metadata(
                handler_call_details.invocation_metadata
            )
        )
        decision = self._auth_decision_engine.evaluate(
            policy=method_policy,
            auth_input=AuthInput(
                token=token,
                signed_metadata_present=signed_metadata_present,
                signed_metadata=signed_metadata,
            ),
        )
        caller_identity = decision.caller_identity
        if not decision.is_allowed or caller_identity is None:
            return _permission_denied_terminator(AUTHENTICATION_FAILED_MESSAGE)

        def authenticated_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            # Store validated auth context for downstream servicer helpers.
            # These attributes are internal-only and must be set exclusively by
            # this interceptor.
            setattr(context, _AUTH_CALLER_IDENTITY_CTX_ATTR, caller_identity)
            if caller_identity.run_id is not None:
                setattr(context, _AUTH_RUN_ID_CTX_ATTR, caller_identity.run_id)
            if token is not None:
                setattr(context, _AUTH_TOKEN_CTX_ATTR, token)
            return unary_unary_handler(request, context)

        return grpc.unary_unary_rpc_method_handler(
            authenticated_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
