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
"""Token authentication interceptors shared by AppIo services."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import APP_TOKEN_HEADER

# Keep a single canonical auth failure message.
# Do not vary details by failure mode (missing token, unknown token, run mismatch,
# malformed metadata, etc.), or callers could use error differences as an auth
# oracle.
_INVALID_TOKEN_DETAILS = "Invalid token."
_AUTH_RUN_ID_CTX_ATTR = "_flwr_appio_authenticated_run_id"
_AUTH_TOKEN_CTX_ATTR = "_flwr_appio_authenticated_token"


def validate_method_requires_token_map(  # pylint: disable=too-many-arguments
    *,
    service_name: str,
    package_name: str,
    rpc_method_names: Sequence[str],
    method_requires_token: Mapping[str, bool],
    table_name: str,
    table_location: str,
) -> None:
    """Validate that token policy table exactly matches service RPCs.

    The table must provide an explicit bool policy (`True` or `False`) for every
    unary RPC in the service. Failing fast at import/startup prevents silently
    exposing newly added RPCs without an auth decision.
    """
    service_fqn = f"{package_name}.{service_name}"
    expected = {f"/{service_fqn}/{rpc_name}" for rpc_name in rpc_method_names}
    configured = set(method_requires_token)
    missing = sorted(expected - configured)
    extra = sorted(configured - expected)
    non_bool_values = sorted(
        method_name
        for method_name, requires_token in method_requires_token.items()
        if not isinstance(requires_token, bool)
    )
    if missing or extra or non_bool_values:
        raise ValueError(
            "Invalid AppIo token policy table.\n"
            f"Table: {table_name}\n"
            f"Location: {table_location}\n"
            f"Service: {service_fqn}\n"
            f"Missing RPC entries: {missing or 'None'}\n"
            f"Unexpected RPC entries: {extra or 'None'}\n"
            f"Entries with non-bool values: {non_bool_values or 'None'}\n"
            "How to fix: update the policy table to include exactly one explicit "
            "bool decision for each RPC exposed by the service."
        )


class _AppTokenState(Protocol):
    """State methods required for AppIo token validation."""

    def get_run_id_by_token(self, token: str) -> int | None:
        """Return run_id for token or None."""

    def verify_token(self, run_id: int, token: str) -> bool:
        """Return whether token is valid for run_id."""


def _abort_invalid_token(context: grpc.ServicerContext) -> None:
    # Use this only when we already have a live ServicerContext inside an
    # executing RPC. This complements `_permission_denied_terminator`, which is
    # used while building the handler in `intercept_service`.
    """Abort current RPC with the canonical invalid token status/details."""
    context.abort(grpc.StatusCode.PERMISSION_DENIED, _INVALID_TOKEN_DETAILS)
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
    """Read App token from invocation metadata."""
    if metadata is None:
        return None
    for key, value in metadata:
        if key != APP_TOKEN_HEADER:
            continue
        if isinstance(value, str):
            return value
        try:
            return value.decode("ascii")
        except UnicodeDecodeError:
            # Malformed/invalid ASCII in metadata: treat as missing/invalid token.
            return None
    return None


def get_authenticated_run_id(context: grpc.ServicerContext) -> int:
    """Get run_id set by AppIo token interceptor."""
    run_id = getattr(context, _AUTH_RUN_ID_CTX_ATTR, None)
    if run_id is None:
        _abort_invalid_token(context)
    return cast(int, run_id)


def get_authenticated_token(context: grpc.ServicerContext) -> str:
    """Get token set by AppIo token interceptor."""
    token = getattr(context, _AUTH_TOKEN_CTX_ATTR, None)
    if token is None:
        _abort_invalid_token(context)
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
        _abort_invalid_token(context)
    return authenticated_run_id


class AppIoTokenAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Add AppIo token metadata to outbound unary RPCs."""

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
        metadata.append((APP_TOKEN_HEADER, self._token))
        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)


class AppIoTokenAuthServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Validate AppIo token metadata based on per-method policy."""

    def __init__(
        self,
        state_provider: Callable[[], _AppTokenState],
        method_requires_token: Mapping[str, bool],
    ) -> None:
        self._state_provider = state_provider
        self._method_requires_token = method_requires_token

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Validate metadata token for methods configured as token-protected."""
        method = handler_call_details.method
        requires_token = self._method_requires_token.get(method, False)
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)

        if not requires_token:
            return method_handler
        if method_handler.unary_unary is None:
            # This interceptor currently protects unary-unary AppIo RPCs only.
            # If method shape is unexpected, fail closed with the same auth
            # error.
            return _permission_denied_terminator(_INVALID_TOKEN_DETAILS)
        unary_unary_handler = cast(
            Callable[[GrpcMessage, grpc.ServicerContext], GrpcMessage],
            method_handler.unary_unary,
        )

        token = _extract_token_from_metadata(handler_call_details.invocation_metadata)
        if token is None:
            return _permission_denied_terminator(_INVALID_TOKEN_DETAILS)

        state = self._state_provider()
        run_id = state.get_run_id_by_token(token)
        if run_id is None or not state.verify_token(run_id, token):
            return _permission_denied_terminator(_INVALID_TOKEN_DETAILS)

        def authenticated_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            # Store validated auth context for downstream servicer helpers.
            # These attributes are internal-only and must be set exclusively by
            # this interceptor.
            setattr(context, _AUTH_RUN_ID_CTX_ATTR, run_id)
            setattr(context, _AUTH_TOKEN_CTX_ATTR, token)
            return unary_unary_handler(request, context)

        return grpc.unary_unary_rpc_method_handler(
            authenticated_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
