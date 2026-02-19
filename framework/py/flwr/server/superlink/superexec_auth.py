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
"""SuperExec authentication helpers for AppIO services."""


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import grpc
import yaml
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common import now
from flwr.common.constant import (
    SUPEREXEC_PUBLIC_KEY_HEADER,
    SUPEREXEC_SIGNATURE_HEADER,
    SUPEREXEC_TIMESTAMP_HEADER,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_TOLERANCE,
    ExecPluginType,
)
from flwr.supercore.primitives.asymmetric import (
    bytes_to_public_key,
    public_key_to_bytes,
    uses_nist_ec_curve,
    verify_signature,
)

_DEFAULT_SUPEREXEC_PLUGINS = {
    ExecPluginType.SERVER_APP,
    ExecPluginType.SIMULATION,
}
_SUPEREXEC_AUTH_HEADERS = {
    SUPEREXEC_PUBLIC_KEY_HEADER,
    SUPEREXEC_SIGNATURE_HEADER,
    SUPEREXEC_TIMESTAMP_HEADER,
}


@dataclass(frozen=True)
class SuperExecAuthConfig:
    """Runtime configuration for SuperExec signed authentication."""

    enabled: bool
    timestamp_tolerance_sec: int
    allowed_public_keys: dict[str, set[bytes]]


def get_disabled_superexec_auth_config() -> SuperExecAuthConfig:
    """Return a disabled SuperExec auth config."""
    return SuperExecAuthConfig(
        enabled=False,
        timestamp_tolerance_sec=TIMESTAMP_TOLERANCE,
        allowed_public_keys={
            ExecPluginType.SERVER_APP: set(),
            ExecPluginType.SIMULATION: set(),
        },
    )


def load_superexec_auth_config(path: str | None) -> SuperExecAuthConfig:
    """Load SuperExec auth config from YAML."""
    if path is None:
        return get_disabled_superexec_auth_config()

    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError("SuperExec auth config must be a YAML mapping.")

    enabled = cfg.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("`enabled` must be a boolean.")

    tolerance = cfg.get("timestamp_tolerance_sec", TIMESTAMP_TOLERANCE)
    if not isinstance(tolerance, int) or tolerance <= 0:
        raise ValueError("`timestamp_tolerance_sec` must be a positive integer.")

    allowed_public_keys: dict[str, set[bytes]] = {
        ExecPluginType.SERVER_APP: set(),
        ExecPluginType.SIMULATION: set(),
    }

    key_entries = cfg.get("public_keys", [])
    if not isinstance(key_entries, list):
        raise ValueError("`public_keys` must be a list.")

    for entry in key_entries:
        public_key_str, allowed_plugins = _parse_public_key_entry(entry)
        public_key_bytes = _parse_and_serialize_public_key(public_key_str)
        for plugin in allowed_plugins:
            allowed_public_keys[plugin].add(public_key_bytes)

    if enabled and not any(allowed_public_keys.values()):
        raise ValueError(
            "SuperExec auth is enabled but no `public_keys` were provided."
        )

    return SuperExecAuthConfig(
        enabled=enabled,
        timestamp_tolerance_sec=tolerance,
        allowed_public_keys=allowed_public_keys,
    )


def superexec_auth_metadata_present(context: grpc.ServicerContext) -> bool:
    """Return True if any SuperExec auth metadata header is present."""
    metadata = dict(context.invocation_metadata())
    return any(header in metadata for header in _SUPEREXEC_AUTH_HEADERS)


# pylint: disable=too-many-locals
def verify_superexec_signed_metadata(
    context: grpc.ServicerContext,
    method: str,
    plugin_type: str,
    cfg: SuperExecAuthConfig,
) -> None:
    """Verify SuperExec signed metadata for an AppIO call."""
    # Require all SuperExec auth headers for protected RPCs.
    metadata = dict(context.invocation_metadata())
    missing_headers = sorted(
        header for header in _SUPEREXEC_AUTH_HEADERS if header not in metadata
    )
    if missing_headers:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Missing SuperExec authentication metadata.",
        )
        raise RuntimeError("Unreachable")

    public_key_bytes = cast(bytes, metadata[SUPEREXEC_PUBLIC_KEY_HEADER])
    signature = cast(bytes, metadata[SUPEREXEC_SIGNATURE_HEADER])
    timestamp_iso = cast(str, metadata[SUPEREXEC_TIMESTAMP_HEADER])

    # Check the caller key is authorized for this plugin type.
    if plugin_type not in cfg.allowed_public_keys:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "SuperExec caller type is not allowed.",
        )
        raise RuntimeError("Unreachable")

    if public_key_bytes not in cfg.allowed_public_keys[plugin_type]:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "SuperExec key is not authorized.",
        )
        raise RuntimeError("Unreachable")

    # Parse and validate key format/curve before signature verification.
    try:
        public_key = bytes_to_public_key(public_key_bytes)
    except ValueError as err:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Invalid SuperExec public key.",
        )
        raise RuntimeError("Unreachable") from err

    if not uses_nist_ec_curve(public_key):
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Invalid SuperExec public key curve.",
        )
        raise RuntimeError("Unreachable")

    # Signature binds both timestamp and method to prevent cross-method replay.
    signed_payload = f"{timestamp_iso}\n{method}".encode()
    if not verify_signature(public_key, signed_payload, signature):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid SuperExec signature.")
        raise RuntimeError("Unreachable")

    try:
        timestamp = datetime.fromisoformat(timestamp_iso)
    except ValueError as err:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid SuperExec timestamp.")
        raise RuntimeError("Unreachable") from err

    # Enforce freshness with a small clock-drift allowance.
    diff_sec = (now() - timestamp).total_seconds()
    min_diff = -SYSTEM_TIME_TOLERANCE
    max_diff = cfg.timestamp_tolerance_sec + SYSTEM_TIME_TOLERANCE
    if not min_diff < diff_sec < max_diff:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Expired SuperExec timestamp.")
        raise RuntimeError("Unreachable")


# pylint: enable=too-many-locals


def _parse_public_key_entry(entry: object) -> tuple[str, set[str]]:
    """Parse one `public_keys` entry into key text and allowed plugin types."""
    # Shorthand form: a plain public key string is accepted for both plugin types.
    if isinstance(entry, str):
        return entry, set(_DEFAULT_SUPEREXEC_PLUGINS)

    # Structured form allows scoping a key to specific plugin types.
    if not isinstance(entry, dict):
        raise ValueError("Each item in `public_keys` must be a string or mapping.")

    public_key_str = entry.get("public_key")
    if not isinstance(public_key_str, str):
        raise ValueError("Each key entry must contain `public_key` as a string.")

    # If plugin scope is omitted, default to all supported SuperExec plugin types.
    allowed_plugins_raw = entry.get("allowed_plugins", list(_DEFAULT_SUPEREXEC_PLUGINS))
    allowed_plugins: set[str]
    if isinstance(allowed_plugins_raw, str):
        allowed_plugins = {allowed_plugins_raw}
    elif isinstance(allowed_plugins_raw, list):
        if not all(isinstance(plugin, str) for plugin in allowed_plugins_raw):
            raise ValueError("`allowed_plugins` must contain only strings.")
        allowed_plugins = set(allowed_plugins_raw)
    else:
        raise ValueError("`allowed_plugins` must be a string or list of strings.")

    if not allowed_plugins:
        raise ValueError("`allowed_plugins` must not be empty.")
    # Reject unknown plugin labels early so config failures are explicit.
    if not allowed_plugins.issubset(_DEFAULT_SUPEREXEC_PLUGINS):
        allowed = ", ".join(sorted(_DEFAULT_SUPEREXEC_PLUGINS))
        raise ValueError(f"`allowed_plugins` must only contain: {allowed}.")

    return public_key_str, allowed_plugins


def _parse_and_serialize_public_key(public_key: str) -> bytes:
    """Parse a configured public key string and return canonical key bytes."""
    key_bytes = public_key.encode("utf-8")
    parsed_key = _load_public_key(key_bytes)
    if not uses_nist_ec_curve(parsed_key):
        raise ValueError("Only NIST EC public keys are supported for SuperExec auth.")
    return public_key_to_bytes(parsed_key)


def _load_public_key(key_bytes: bytes) -> ec.EllipticCurvePublicKey:
    """Load an EC public key from SSH or PEM encoded bytes."""
    for loader in (
        serialization.load_ssh_public_key,
        serialization.load_pem_public_key,
    ):
        try:
            key = loader(key_bytes)
            if isinstance(key, ec.EllipticCurvePublicKey):
                return key
        except (ValueError, UnsupportedAlgorithm):
            continue
    raise ValueError("Unable to parse SuperExec public key.")
