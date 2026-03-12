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
"""Policy types and validation helpers for AppIo authentication."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constant import AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA, AUTH_MECHANISM_TOKEN


@dataclass(frozen=True)
class MethodAuthPolicy:
    """Authentication policy for a single RPC method.

    Policy is intentionally separate from mechanism implementations. This keeps RPC
    access decisions declarative and lets mechanisms evolve independently.

    This policy currently supports at most one mechanism per RPC.
    `requires_run_id_match` is enforced in transport/handler layers that
    have request context (for example, gRPC interceptors/servicers), not by
    `AuthDecisionEngine`.
    """

    required_mechanism: str | None = None
    requires_run_id_match: bool = False

    def __post_init__(self) -> None:
        """Validate cross-field invariants."""
        if self.required_mechanism is None and self.requires_run_id_match:
            raise ValueError(
                "requires_run_id_match=True requires a non-None required_mechanism."
            )

    @property
    def requires_authentication(self) -> bool:
        """Return whether authentication is required for this method."""
        return self.required_mechanism is not None

    @classmethod
    def no_auth(cls) -> "MethodAuthPolicy":
        """Create a policy for methods that do not require authentication."""
        return cls(required_mechanism=None)

    @classmethod
    def token_required(
        cls, *, requires_run_id_match: bool = False
    ) -> "MethodAuthPolicy":
        """Create a policy for methods requiring token authentication."""
        return cls(
            required_mechanism=AUTH_MECHANISM_TOKEN,
            requires_run_id_match=requires_run_id_match,
        )

    @classmethod
    def signed_metadata_required(
        cls, *, requires_run_id_match: bool = False
    ) -> "MethodAuthPolicy":
        """Create a policy for methods requiring signed metadata authentication."""
        return cls(
            required_mechanism=AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA,
            requires_run_id_match=requires_run_id_match,
        )


# Keep explicit keyword arguments for clearer startup error messages.
def validate_method_auth_policy_map(  # pylint: disable=too-many-arguments
    *,
    service_name: str,
    package_name: str,
    rpc_method_names: Sequence[str],
    method_auth_policy: Mapping[str, MethodAuthPolicy],
    table_name: str,
    table_location: str,
) -> None:
    """Validate that method auth policy table exactly matches service RPCs."""
    service_fqn = f"{package_name}.{service_name}"
    expected = {f"/{service_fqn}/{rpc_name}" for rpc_name in rpc_method_names}
    configured = set(method_auth_policy)
    missing = sorted(expected - configured)
    extra = sorted(configured - expected)
    invalid_policy_values = sorted(
        method_name
        for method_name, policy in method_auth_policy.items()
        if not isinstance(policy, MethodAuthPolicy)
    )
    if missing or extra or invalid_policy_values:
        raise ValueError(
            "Invalid AppIo method auth policy table.\n"
            f"Table: {table_name}\n"
            f"Location: {table_location}\n"
            f"Service: {service_fqn}\n"
            f"Missing RPC entries: {missing or 'None'}\n"
            f"Unexpected RPC entries: {extra or 'None'}\n"
            f"Entries with invalid policy objects: {invalid_policy_values or 'None'}\n"
            "How to fix: update the policy table to include exactly one "
            "`MethodAuthPolicy` entry for each RPC exposed by the service."
        )
