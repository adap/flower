# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""SQLite based implementation of server authentication state."""

from typing import Set

from flwr.server.state.authentication.authentication_state import AuthenticationState
from flwr.server.state.sqlite_state import SqliteState


class SqliteAuthState(AuthenticationState, SqliteState):
    """SQLite-based authentication state implementation."""

    def store_node_id_public_key_pair(self, node_id: int, public_key: bytes) -> None:
        """Store `node_id` and `public_key` as key-value pair in state."""
        query = (
            "INSERT OR REPLACE INTO node_key (node_id, public_key) "
            "VALUES (:node_id, :public_key)"
        )
        self.query(query, {"node_id": node_id, "public_key": public_key})

    def get_public_key_from_node_id(self, node_id: int) -> bytes:
        """Get client's public key in urlsafe bytes for `node_id`."""
        query = "SELECT public_key FROM node_key WHERE node_id = :node_id"
        rows = self.query(query, {"node_id": node_id})
        public_key: bytes = rows[0]["public_key"]
        return public_key

    def store_server_public_private_key(
        self, public_key: bytes, private_key: bytes
    ) -> None:
        """Store server's `public_key` and `private_key` in state."""
        query = (
            "INSERT OR REPLACE INTO credential (public_key, private_key) "
            "VALUES (:public_key, :private_key)"
        )
        self.query(query, {"public_key": public_key, "private_key": private_key})

    def get_server_private_key(self) -> bytes:
        """Get server private key in urlsafe bytes."""
        query = "SELECT private_key FROM credential"
        rows = self.query(query)
        private_key: bytes = rows[0]["private_key"]
        return private_key

    def get_server_public_key(self) -> bytes:
        """Get server public key in urlsafe bytes."""
        query = "SELECT public_key FROM credential"
        rows = self.query(query)
        public_key: bytes = rows[0]["public_key"]
        return public_key

    def store_client_public_keys(self, public_keys: Set[bytes]) -> None:
        """Store a set of client public keys in state."""
        query = "INSERT INTO public_key (public_key) VALUES (:public_key)"
        for public_key in public_keys:
            self.query(query, {"public_key": public_key})

    def store_client_public_key(self, public_key: bytes) -> None:
        """Retrieve a client public key in state."""
        query = "INSERT INTO public_key (public_key) VALUES (:public_key)"
        self.query(query, {"public_key": public_key})

    def get_client_public_keys(self) -> Set[bytes]:
        """Retrieve all currently stored client public keys as a set."""
        query = "SELECT public_key FROM public_key"
        rows = self.query(query)
        result: Set[bytes] = {row["public_key"] for row in rows}
        return result
