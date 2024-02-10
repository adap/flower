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
"""In-memory Authentication State implementation."""

from typing import Dict, Set

from authentication_state import AuthenticationState
from in_memory_state import InMemoryState


class InMemoryAuthState(AuthenticationState, InMemoryState):
    def __init__(self) -> None:
        """Init InMemoryAuthState."""
        super().__init__()
        self.node_id_public_key_dict: Dict[int, bytes] = {}
        self.client_public_keys: Set[bytes] = set()
        self.server_public_key: bytes = b""
        self.server_private_key: bytes = b""

    def store_node_id_public_key_pair(self, node_id: int, public_key: bytes) -> None:
        """Store `node_id` and `public_key` as key-value pair in state."""
        if node_id not in self.node_ids:
            raise ValueError(f"Node {node_id} not found")
        if node_id in self.node_id_public_key_dict:
            raise ValueError(f"Node {node_id} has already assigned a public key")
        self.node_id_public_key_dict[node_id] = public_key

    def get_public_key_from_node_id(self, node_id: int) -> bytes:
        """Get client's public key in urlsafe bytes for `node_id`."""
        if node_id in self.node_id_public_key_dict:
            return self.node_id_public_key_dict[node_id]
        return b""

    def store_server_public_private_key(
        self, public_key: bytes, private_key: bytes
    ) -> None:
        """Store server's `public_key` and `private_key` in state."""
        self.server_private_key = private_key
        self.server_public_key = public_key

    def get_server_private_key(self) -> bytes:
        """Get server private key in urlsafe bytes."""
        return self.server_private_key

    def get_server_public_key(self) -> bytes:
        """Get server public key in urlsafe bytes."""
        return self.server_public_key

    def store_client_public_keys(self, public_keys: Set[bytes]) -> None:
        """Store a set of client public keys in state."""
        self.client_public_keys = public_keys

    def store_client_public_key(self, public_key: bytes) -> None:
        """Retrieve a client public key in state."""
        self.client_public_keys.add(public_key)

    def get_client_public_keys(self) -> Set[bytes]:
        """Retrieve all currently stored client public keys as a set."""
        return self.client_public_keys
