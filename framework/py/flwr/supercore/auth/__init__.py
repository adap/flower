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
"""Transport-agnostic authentication primitives for AppIo services."""

from .appio_auth import (
    AuthDecision,
    AuthDecisionFailureReason,
    AuthDecisionEngine,
    AuthInput,
    Authenticator,
    CallerIdentity,
    SignedMetadataAuthInput,
    TokenAuthenticator,
)
from .constant import (
    AuthSelectionMode,
    APPIO_SIGNED_METADATA_METHOD_HEADER,
    APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER,
    APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER,
    APPIO_SIGNED_METADATA_SIGNATURE_HEADER,
    APPIO_SIGNED_METADATA_TIMESTAMP_HEADER,
    CALLER_TYPE_APP_EXECUTOR,
    CALLER_TYPE_SUPEREXEC,
    AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA,
    AUTH_MECHANISM_TOKEN,
    AUTH_SELECTION_MODE_ANY_ONE,
    AUTH_SELECTION_MODE_EXACTLY_ONE,
)
from .policy import MethodAuthPolicy, validate_method_auth_policy_map

__all__ = [
    "AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA",
    "AUTH_MECHANISM_TOKEN",
    "AuthSelectionMode",
    "APPIO_SIGNED_METADATA_METHOD_HEADER",
    "APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER",
    "APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER",
    "APPIO_SIGNED_METADATA_SIGNATURE_HEADER",
    "APPIO_SIGNED_METADATA_TIMESTAMP_HEADER",
    "CALLER_TYPE_APP_EXECUTOR",
    "CALLER_TYPE_SUPEREXEC",
    "AUTH_SELECTION_MODE_ANY_ONE",
    "AUTH_SELECTION_MODE_EXACTLY_ONE",
    "AuthDecision",
    "AuthDecisionFailureReason",
    "AuthDecisionEngine",
    "AuthInput",
    "Authenticator",
    "CallerIdentity",
    "MethodAuthPolicy",
    "SignedMetadataAuthInput",
    "TokenAuthenticator",
    "validate_method_auth_policy_map",
]
