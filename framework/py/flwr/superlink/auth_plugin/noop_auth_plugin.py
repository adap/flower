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
"""Concrete NoOp implementation for Servicer-side account authentication and
authorization plugins."""


from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID, AuthType
from flwr.common.typing import AccountInfo, UserAuthCredentials, UserAuthLoginDetails

from .auth_plugin import ControlAuthPlugin, ControlAuthzPlugin

NOOP_ACCOUNT_INFO = AccountInfo(
    flwr_aid=NOOP_FLWR_AID,
    account_name=NOOP_ACCOUNT_NAME,
)


class NoOpControlAuthPlugin(ControlAuthPlugin):
    """No-operation implementation of ControlAuthPlugin."""

    def __init__(
        self,
        user_auth_config_path: Path,
        verify_tls_cert: bool,
    ):
        pass

    def get_login_details(self) -> Optional[UserAuthLoginDetails]:
        """Get the login details."""
        # This allows the `flwr login` command to load the NoOp plugin accordingly,
        # which then raises a LoginError when attempting to login.
        return UserAuthLoginDetails(
            auth_type=AuthType.NOOP,  # No operation auth type
            device_code="",
            verification_uri_complete="",
            expires_in=0,
            interval=0,
        )

    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> tuple[bool, Optional[AccountInfo]]:
        """Return valid for no-op plugin."""
        return True, NOOP_ACCOUNT_INFO

    def get_auth_tokens(self, device_code: str) -> Optional[UserAuthCredentials]:
        """Get authentication tokens."""
        raise RuntimeError("NoOp plugin does not support getting auth tokens.")

    def refresh_tokens(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> tuple[
        Optional[Sequence[tuple[str, Union[str, bytes]]]], Optional[AccountInfo]
    ]:
        """Refresh authentication tokens in the provided metadata."""
        return metadata, NOOP_ACCOUNT_INFO


class NoOpControlAuthzPlugin(ControlAuthzPlugin):
    """No-operation implementation of ControlAuthzPlugin."""

    def __init__(self, user_auth_config_path: Path, verify_tls_cert: bool):
        pass

    def verify_user_authorization(self, account_info: AccountInfo) -> bool:
        """Return True for no-op plugin."""
        return True
