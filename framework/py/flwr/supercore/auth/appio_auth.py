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
"""Transport-agnostic authentication primitives used by AppIo adapters."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .constant import (
    AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA,
    AUTH_MECHANISM_TOKEN,
    CALLER_TYPE_APP_EXECUTOR,
)
from .policy import MethodAuthPolicy


@dataclass(frozen=True)
class SignedMetadataAuthInput:
    """Signed metadata payload extracted from request metadata.

    This is transport-normalized input only. Signature/timestamp verification remains in
    authenticator implementations so policy logic stays mechanism-agnostic. This payload
    is raw metadata and is not pre-verified at extraction time.
    """

    # Caller's public key from metadata, used for key identity + signature verify.
    public_key: bytes
    # Signature over the expected payload (for example, timestamp + method).
    signature: bytes
    # Caller-provided ISO timestamp used for freshness and replay-window checks.
    timestamp_iso: str
    # RPC method name bound into signature payload to prevent cross-method replay.
    method: str
    # Expected SuperExec plugin scope (for allowlist/policy checks).
    plugin_type: str | None = None


@dataclass(frozen=True)
class AuthInput:
    """Authentication data extracted from a transport-specific request.

    ``AuthInput`` is the single handoff object from transport adapters to the auth
    layer. Keeping all optional inputs here makes it easy to add mechanisms
    without changing policy or interceptor call signatures.
    """

    token: str | None = None
    # True means signed-metadata auth material was supplied on the request path.
    # This can be True while ``signed_metadata`` is None when extraction sees a
    # partial/malformed signed-metadata payload.
    signed_metadata_present: bool = False
    signed_metadata: SignedMetadataAuthInput | None = None

    def __post_init__(self) -> None:
        """Validate signed metadata presence invariants."""
        if self.signed_metadata is not None and not self.signed_metadata_present:
            raise ValueError(
                "signed_metadata_present must be True when signed_metadata is set."
            )


@dataclass(frozen=True)
class CallerIdentity:
    """Normalized authenticated caller identity.

    This shape supports both app-executor and SuperExec callers. Fields are
    intentionally optional so one identity type can represent multiple auth mechanisms.
    """

    # Auth mechanism that produced this identity (token, signed-metadata, ...).
    mechanism: str
    # Normalized caller category (for example, app_executor or superexec).
    caller_type: str
    # Authenticated run binding when applicable; None for non-run-bound callers.
    run_id: int | None = None
    # Stable key identifier for key-based callers; None for non-key mechanisms.
    key_fingerprint: str | None = None


@dataclass(frozen=True)
class AuthDecision:
    """Result of evaluating an ``AuthInput`` against a method policy.

    ``failure_reason`` is internal-only for tests/diagnostics. Interceptors still
    map denials to canonical external responses (for example, PERMISSION_DENIED).
    """

    is_allowed: bool
    caller_identity: CallerIdentity | None
    failure_reason: "AuthDecisionFailureReason | None" = None


class AuthDecisionFailureReason(Enum):
    """Internal reasons for auth denials."""

    MISSING_AUTH_INPUT = "missing_auth_input"
    INVALID_AUTH_INPUT = "invalid_auth_input"
    NON_REQUIRED_MECHANISM_PRESENT = "non_required_mechanism_present"
    POLICY_MISCONFIGURED = "policy_misconfigured"


class Authenticator(Protocol):
    """Authentication primitive for one mechanism."""

    mechanism: str

    def is_present(self, auth_input: AuthInput) -> bool:
        """Return whether this mechanism's auth input is present."""

    def authenticate(self, auth_input: AuthInput) -> CallerIdentity | None:
        """Return caller identity if authentication succeeds."""


class AuthDecisionEngine:
    """Evaluate method policy against available authenticators.

    The engine is transport-independent and currently enforces one configured mechanism
    per RPC. It decides mechanism compatibility and delegates cryptographic/token checks
    to authenticators.
    """

    def __init__(
        self,
        authenticators: Mapping[str, Authenticator],
        method_auth_policies: Mapping[str, MethodAuthPolicy],
    ) -> None:
        self._authenticators = authenticators
        # Validate at construction to fail fast on startup configuration bugs.
        self._validate_policy_mechanisms(method_auth_policies)

    def _validate_policy_mechanisms(
        self, method_auth_policies: Mapping[str, MethodAuthPolicy]
    ) -> None:
        """Fail fast if policy references unknown mechanisms."""
        invalid_policy_values: list[str] = []
        missing_by_method: dict[str, list[str]] = {}
        for method, policy in method_auth_policies.items():
            if not isinstance(policy, MethodAuthPolicy):
                invalid_policy_values.append(method)
                continue
            required_mechanism = policy.required_mechanism
            if (
                required_mechanism is not None
                and required_mechanism not in self._authenticators
            ):
                missing_by_method[method] = [required_mechanism]
        if invalid_policy_values or missing_by_method:
            invalid_entries = invalid_policy_values or "None"
            missing_entries = missing_by_method or "None"
            raise ValueError(
                "Invalid method auth policies for AuthDecisionEngine.\n"
                f"Entries with invalid policy objects: {invalid_entries}\n"
                "Entries referencing mechanisms without authenticators: "
                f"{missing_entries}"
            )

    @staticmethod
    def _present_mechanisms_from_auth_input(auth_input: AuthInput) -> set[str]:
        """Return present mechanisms based directly on normalized auth input."""
        present_mechanisms: set[str] = set()
        if auth_input.token is not None:
            present_mechanisms.add(AUTH_MECHANISM_TOKEN)
        if auth_input.signed_metadata_present:
            present_mechanisms.add(AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA)
        return present_mechanisms

    def evaluate(self, policy: MethodAuthPolicy, auth_input: AuthInput) -> AuthDecision:
        """Evaluate authentication for a single method invocation."""
        if not policy.requires_authentication:
            return AuthDecision(is_allowed=True, caller_identity=None)

        required_mechanism = policy.required_mechanism
        required_authenticator = (
            None
            if required_mechanism is None
            else self._authenticators.get(required_mechanism)
        )
        failure_reason: AuthDecisionFailureReason | None = None

        if required_mechanism is None or required_authenticator is None:
            # Defensive fallback for malformed policies and runtime safety.
            failure_reason = AuthDecisionFailureReason.POLICY_MISCONFIGURED
        else:
            present_mechanisms = self._present_mechanisms_from_auth_input(auth_input)

            # Check if the required mechanism is missing.
            if required_mechanism not in present_mechanisms:
                failure_reason = AuthDecisionFailureReason.MISSING_AUTH_INPUT
            # Check if any present mechanism is not the one required by policy.
            # This is explicitly denied to keep one-mechanism-per-RPC semantics.
            elif any(
                mechanism != required_mechanism for mechanism in present_mechanisms
            ):
                failure_reason = (
                    AuthDecisionFailureReason.NON_REQUIRED_MECHANISM_PRESENT
                )
            # If required mechanism is present and no extra mechanism is present,
            # attempt authentication with that mechanism.
            else:
                caller_identity = required_authenticator.authenticate(auth_input)
                if caller_identity is not None:
                    return AuthDecision(
                        is_allowed=True,
                        caller_identity=caller_identity,
                        failure_reason=None,
                    )
                failure_reason = AuthDecisionFailureReason.INVALID_AUTH_INPUT

        return AuthDecision(
            is_allowed=False,
            caller_identity=None,
            failure_reason=failure_reason,
        )


class _TokenState(Protocol):
    """State methods required for token authentication."""

    def get_run_id_by_token(self, token: str) -> int | None:
        """Return run_id for token or None."""

    def verify_token(self, run_id: int, token: str) -> bool:
        """Return whether token is valid for run_id."""


class TokenAuthenticator:
    """Token-based authenticator for AppIo callers.

    This is one concrete mechanism implementation registered into the decision engine.
    Future SuperExec signed-metadata auth should follow the same pattern.
    """

    mechanism = AUTH_MECHANISM_TOKEN

    def __init__(self, state_provider: Callable[[], _TokenState]) -> None:
        self._state_provider = state_provider

    def is_present(self, auth_input: AuthInput) -> bool:
        """Return whether token auth input is present."""
        return auth_input.token is not None

    def authenticate(self, auth_input: AuthInput) -> CallerIdentity | None:
        """Authenticate caller using AppIo token."""
        token = auth_input.token
        if token is None:
            return None

        state = self._state_provider()
        run_id = state.get_run_id_by_token(token)
        if run_id is None or not state.verify_token(run_id, token):
            return None

        return CallerIdentity(
            mechanism=self.mechanism,
            caller_type=CALLER_TYPE_APP_EXECUTOR,
            run_id=run_id,
        )
