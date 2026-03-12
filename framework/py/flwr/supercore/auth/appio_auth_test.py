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
"""Tests for transport-agnostic AppIo auth primitives and policy logic."""

from typing import cast
from unittest import TestCase
from unittest.mock import Mock

from flwr.supercore.auth.appio_auth import (
    AuthDecisionEngine,
    AuthDecisionFailureReason,
    AuthInput,
    CallerIdentity,
    SignedMetadataAuthInput,
    TokenAuthenticator,
)
from flwr.supercore.auth.constant import (
    AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA,
    AUTH_MECHANISM_TOKEN,
    CALLER_TYPE_APP_EXECUTOR,
    CALLER_TYPE_SUPEREXEC,
)
from flwr.supercore.auth.policy import MethodAuthPolicy, validate_method_auth_policy_map


class _SignedMetadataPresenceAuthenticator:
    """Test authenticator using signed metadata presence only."""

    mechanism = AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA

    def is_present(self, auth_input: AuthInput) -> bool:
        """Return whether signed metadata auth input is present."""
        return auth_input.signed_metadata_present

    def authenticate(self, auth_input: AuthInput) -> CallerIdentity | None:
        """Return synthetic identity when signed metadata is fully populated."""
        if auth_input.signed_metadata is None:
            return None
        return CallerIdentity(
            mechanism=self.mechanism,
            caller_type=CALLER_TYPE_SUPEREXEC,
            key_fingerprint="test-fingerprint",
        )


class TestAuthDecisionEngine(TestCase):
    """Unit tests for ``AuthDecisionEngine``."""

    def test_no_auth_policy_always_allows(self) -> None:
        """Methods with no auth policy are always allowed."""
        engine = AuthDecisionEngine(authenticators={}, method_auth_policy={})

        decision = engine.evaluate(
            policy=MethodAuthPolicy.no_auth(),
            auth_input=AuthInput(token=None),
        )

        self.assertTrue(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertIsNone(decision.failure_reason)

    def test_token_policy_allows_with_matching_authenticator(self) -> None:
        """Token policy succeeds when token authenticator yields identity."""
        state = Mock()
        state.get_run_id_by_token.return_value = 13
        state.verify_token.return_value = True
        engine = AuthDecisionEngine(
            authenticators={AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state)},
            method_auth_policy={},
        )

        decision = engine.evaluate(
            policy=MethodAuthPolicy.token_required(),
            auth_input=AuthInput(token="valid-token"),
        )

        self.assertTrue(decision.is_allowed)
        self.assertEqual(
            decision.caller_identity,
            CallerIdentity(
                mechanism=AUTH_MECHANISM_TOKEN,
                caller_type=CALLER_TYPE_APP_EXECUTOR,
                run_id=13,
            ),
        )
        self.assertIsNone(decision.failure_reason)

    def test_engine_fails_fast_when_policy_references_missing_authenticator(
        self,
    ) -> None:
        """Construction fails if any required mechanism has no authenticator."""
        with self.assertRaisesRegex(
            ValueError, "references mechanisms without authenticators"
        ):
            AuthDecisionEngine(
                authenticators={},
                method_auth_policy={
                    "/flwr.proto.ServerAppIo/GetNodes": (
                        MethodAuthPolicy.token_required()
                    )
                },
            )

    def test_token_policy_denies_when_authenticator_missing(self) -> None:
        """Policy requiring an unavailable authenticator is denied."""
        engine = AuthDecisionEngine(authenticators={}, method_auth_policy={})

        decision = engine.evaluate(
            policy=MethodAuthPolicy.token_required(),
            auth_input=AuthInput(token="token"),
        )

        self.assertFalse(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertEqual(
            decision.failure_reason, AuthDecisionFailureReason.POLICY_MISCONFIGURED
        )

    def test_token_policy_denies_when_token_missing(self) -> None:
        """Token policy requires token input to be present."""
        state = Mock()
        engine = AuthDecisionEngine(
            authenticators={AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state)},
            method_auth_policy={},
        )

        decision = engine.evaluate(
            policy=MethodAuthPolicy.token_required(),
            auth_input=AuthInput(token=None),
        )

        self.assertFalse(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertEqual(
            decision.failure_reason, AuthDecisionFailureReason.MISSING_AUTH_INPUT
        )

    def test_token_required_denies_when_extra_signed_metadata_is_present(self) -> None:
        """Required token auth denies when another mechanism is simultaneously
        present."""
        state = Mock()
        state.get_run_id_by_token.return_value = 1
        state.verify_token.return_value = True
        engine = AuthDecisionEngine(
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: (
                    _SignedMetadataPresenceAuthenticator()
                ),
            },
            method_auth_policy={},
        )

        decision = engine.evaluate(
            policy=MethodAuthPolicy.token_required(),
            auth_input=AuthInput(
                token="valid-token",
                signed_metadata=SignedMetadataAuthInput(
                    public_key=b"pk",
                    signature=b"sig",
                    timestamp_iso="2026-03-09T10:00:00",
                    method="/flwr.proto.ServerAppIo/GetNodes",
                ),
                signed_metadata_present=True,
            ),
        )

        self.assertFalse(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertEqual(
            decision.failure_reason,
            AuthDecisionFailureReason.INVALID_MECHANISM_COMBINATION,
        )

    def test_signed_metadata_required_denies_when_extra_token_is_present(self) -> None:
        """Required signed-metadata auth denies when token input is also present."""
        state = Mock()
        engine = AuthDecisionEngine(
            authenticators={
                AUTH_MECHANISM_TOKEN: TokenAuthenticator(lambda: state),
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: (
                    _SignedMetadataPresenceAuthenticator()
                ),
            },
            method_auth_policy={},
        )

        decision = engine.evaluate(
            policy=MethodAuthPolicy.signed_metadata_required(),
            auth_input=AuthInput(
                token="some-token",
                signed_metadata=SignedMetadataAuthInput(
                    public_key=b"pk",
                    signature=b"sig",
                    timestamp_iso="2026-03-09T10:00:00",
                    method="/flwr.proto.ServerAppIo/GetNodes",
                ),
                signed_metadata_present=True,
            ),
        )

        self.assertFalse(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertEqual(
            decision.failure_reason,
            AuthDecisionFailureReason.INVALID_MECHANISM_COMBINATION,
        )

    def test_signed_metadata_required_denies_when_input_is_malformed(self) -> None:
        """Malformed signed metadata is invalid input, not missing input."""
        engine = AuthDecisionEngine(
            authenticators={
                AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: (
                    _SignedMetadataPresenceAuthenticator()
                )
            },
            method_auth_policy={},
        )

        decision = engine.evaluate(
            policy=MethodAuthPolicy.signed_metadata_required(),
            auth_input=AuthInput(signed_metadata_present=True, signed_metadata=None),
        )

        self.assertFalse(decision.is_allowed)
        self.assertIsNone(decision.caller_identity)
        self.assertEqual(
            decision.failure_reason,
            AuthDecisionFailureReason.INVALID_AUTH_INPUT,
        )


class TestTokenAuthenticator(TestCase):
    """Unit tests for ``TokenAuthenticator``."""

    def test_missing_token_is_denied(self) -> None:
        """No token in input returns no identity."""
        state = Mock()
        authenticator = TokenAuthenticator(lambda: state)

        self.assertFalse(authenticator.is_present(AuthInput(token=None)))
        caller_identity = authenticator.authenticate(AuthInput(token=None))

        self.assertIsNone(caller_identity)
        state.get_run_id_by_token.assert_not_called()

    def test_invalid_token_is_denied(self) -> None:
        """Unknown token returns no identity."""
        state = Mock()
        state.get_run_id_by_token.return_value = None
        authenticator = TokenAuthenticator(lambda: state)

        self.assertTrue(authenticator.is_present(AuthInput(token="invalid-token")))
        caller_identity = authenticator.authenticate(AuthInput(token="invalid-token"))

        self.assertIsNone(caller_identity)
        state.get_run_id_by_token.assert_called_once_with("invalid-token")
        state.verify_token.assert_not_called()

    def test_valid_token_returns_identity(self) -> None:
        """Valid token returns a normalized caller identity."""
        state = Mock()
        state.get_run_id_by_token.return_value = 42
        state.verify_token.return_value = True
        authenticator = TokenAuthenticator(lambda: state)

        caller_identity = authenticator.authenticate(AuthInput(token="valid-token"))

        self.assertEqual(
            caller_identity,
            CallerIdentity(
                mechanism=AUTH_MECHANISM_TOKEN,
                caller_type=CALLER_TYPE_APP_EXECUTOR,
                run_id=42,
            ),
        )
        state.get_run_id_by_token.assert_called_once_with("valid-token")
        state.verify_token.assert_called_once_with(42, "valid-token")


class TestSignedMetadataPresence(TestCase):
    """Unit tests for signed metadata presence detection hooks."""

    def test_signed_metadata_presence_detected(self) -> None:
        """Signed metadata input should be detectable by a dedicated authenticator."""
        authenticator = _SignedMetadataPresenceAuthenticator()

        self.assertTrue(
            authenticator.is_present(
                AuthInput(
                    signed_metadata_present=True,
                    signed_metadata=SignedMetadataAuthInput(
                        public_key=b"pk",
                        signature=b"sig",
                        timestamp_iso="2026-03-09T10:00:00",
                        method="/flwr.proto.ServerAppIo/GetNodes",
                    ),
                )
            )
        )

    def test_signed_metadata_absence_detected(self) -> None:
        """Missing signed metadata should not appear as present."""
        authenticator = _SignedMetadataPresenceAuthenticator()

        self.assertFalse(
            authenticator.is_present(AuthInput(signed_metadata_present=False))
        )

    def test_signed_metadata_partial_payload_detected_as_present(self) -> None:
        """Presence flag distinguishes malformed input from a missing mechanism."""
        authenticator = _SignedMetadataPresenceAuthenticator()

        self.assertTrue(
            authenticator.is_present(AuthInput(signed_metadata_present=True))
        )


class TestAuthInputInvariant(TestCase):
    """Unit tests for ``AuthInput`` invariants."""

    def test_signed_metadata_requires_presence_flag(self) -> None:
        """Setting signed metadata without presence flag should fail."""
        with self.assertRaisesRegex(ValueError, "signed_metadata_present must be True"):
            AuthInput(
                signed_metadata=SignedMetadataAuthInput(
                    public_key=b"pk",
                    signature=b"sig",
                    timestamp_iso="2026-03-09T10:00:00",
                    method="/flwr.proto.ServerAppIo/GetNodes",
                )
            )


class TestMethodAuthPolicyValidation(TestCase):
    """Unit tests for method policy table validation."""

    def test_validate_method_auth_policy_map_accepts_matching_table(self) -> None:
        """Validation passes when table exactly matches service RPC names."""
        table = {
            "/flwr.proto.TestService/Foo": MethodAuthPolicy.no_auth(),
            "/flwr.proto.TestService/Bar": MethodAuthPolicy.token_required(),
        }

        validate_method_auth_policy_map(
            service_name="TestService",
            package_name="flwr.proto",
            rpc_method_names=("Foo", "Bar"),
            method_auth_policy=table,
            table_name="TEST_POLICY",
            table_location=__file__,
        )

    def test_validate_method_auth_policy_map_rejects_invalid_table(self) -> None:
        """Validation fails when policy table misses or mis-types entries."""
        with self.assertRaisesRegex(
            ValueError, "Invalid AppIo method auth policy table."
        ):
            validate_method_auth_policy_map(
                service_name="TestService",
                package_name="flwr.proto",
                rpc_method_names=("Foo",),
                method_auth_policy=cast(
                    dict[str, MethodAuthPolicy],
                    {
                        "/flwr.proto.TestService/Bar": MethodAuthPolicy.no_auth(),
                        "/flwr.proto.TestService/Foo": "bad-policy",
                    },
                ),
                table_name="TEST_POLICY",
                table_location=__file__,
            )
