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
"""File-based credential store implementation."""


import base64
from pathlib import Path
from typing import cast

import yaml

from ..utils import get_flwr_home
from .credential_store import CredentialStore

CREDENTIAL_FILE_PATH = get_flwr_home() / "credentials.yaml"


class FileCredentialStore(CredentialStore):
    """File-based credential store implementation."""

    def __init__(self, file_path: Path | None = None) -> None:
        """Initialize the file credential store.

        Parameters
        ----------
        file_path : Path | None
            Path to the credentials file. If None, uses default path.
        """
        self.file_path = file_path or CREDENTIAL_FILE_PATH

    def _load_credentials(self) -> dict[str, str]:
        """Load credentials from file."""
        if not self.file_path.exists():
            return {}
        with self.file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return cast(dict[str, str], data) if data else {}

    def _save_credentials(self, credentials: dict[str, str]) -> None:
        """Save credentials to file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(credentials, f)

    def set(self, key: str, value: bytes) -> None:
        """Set a credential in the store."""
        credentials = self._load_credentials()
        credentials[key] = base64.b64encode(value).decode("utf-8")
        self._save_credentials(credentials)

    def get(self, key: str) -> bytes | None:
        """Get a credential from the store."""
        credentials = self._load_credentials()
        encoded_value = credentials.get(key)
        if encoded_value is None:
            return None
        return base64.b64decode(encoded_value)

    def delete(self, key: str) -> None:
        """Delete a credential from the store."""
        credentials = self._load_credentials()
        if key in credentials:
            del credentials[key]
            self._save_credentials(credentials)
