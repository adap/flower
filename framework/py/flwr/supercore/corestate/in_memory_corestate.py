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
"""In-memory CoreState implementation."""


import secrets
from threading import Lock

from flwr.common.constant import FLWR_APP_TOKEN_LENGTH

from .corestate import CoreState


class InMemoryCoreState(CoreState):
    """In-memory CoreState implementation."""

    def __init__(self) -> None:
        # Store run ID to token mapping and token to run ID mapping
        self.token_store: dict[int, str] = {}
        self.token_to_run_id: dict[str, int] = {}
        self.lock_token_store = Lock()

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        with self.lock_token_store:
            if run_id in self.token_store:
                return None  # Token already created for this run ID
            self.token_store[run_id] = token
            self.token_to_run_id[token] = run_id
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        with self.lock_token_store:
            return self.token_store.get(run_id) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        with self.lock_token_store:
            token = self.token_store.pop(run_id, None)
            if token is not None:
                self.token_to_run_id.pop(token, None)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        with self.lock_token_store:
            return self.token_to_run_id.get(token)

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        raise NotImplementedError
