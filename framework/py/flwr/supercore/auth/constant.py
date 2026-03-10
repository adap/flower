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
"""Auth-related constants shared by AppIo auth abstractions."""

from typing import Final, Literal, TypeAlias

AUTH_MECHANISM_TOKEN: Final[str] = "token"
AUTH_MECHANISM_SUPEREXEC_SIGNED_METADATA: Final[str] = "superexec-signed-metadata"
CALLER_TYPE_APP_EXECUTOR: Final[str] = "app_executor"
CALLER_TYPE_SUPEREXEC: Final[str] = "superexec"
AUTHENTICATION_FAILED_MESSAGE: Final[str] = "Authentication failed."

AuthSelectionMode: TypeAlias = Literal["any_one", "exactly_one"]
AUTH_SELECTION_MODE_ANY_ONE: Final[AuthSelectionMode] = "any_one"
AUTH_SELECTION_MODE_EXACTLY_ONE: Final[AuthSelectionMode] = "exactly_one"

# gRPC metadata keys for signed-metadata AppIo auth input extraction.
APP_TOKEN_HEADER: Final[str] = "flwr-app-token"
APPIO_SIGNED_METADATA_PUBLIC_KEY_HEADER: Final[str] = "flwr-superexec-public-key-bin"
APPIO_SIGNED_METADATA_SIGNATURE_HEADER: Final[str] = "flwr-superexec-signature-bin"
APPIO_SIGNED_METADATA_TIMESTAMP_HEADER: Final[str] = "flwr-superexec-timestamp"
APPIO_SIGNED_METADATA_METHOD_HEADER: Final[str] = "flwr-superexec-method"
APPIO_SIGNED_METADATA_PLUGIN_TYPE_HEADER: Final[str] = "flwr-superexec-plugin-type"
