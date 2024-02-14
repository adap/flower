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
"""Abstract base class AuthenticationState."""

import abc
from typing import Set

from flwr.server.superlink.state import State


class AuthenticationState(State, abc.ABC):
    """Abstract Authentication State."""

    @abc.abstractmethod
    def store_node_id_public_key_pair(self, node_id: int, public_key: bytes) -> None:
        """Store `node_id` and `public_key` as key-value pair in state."""

    @abc.abstractmethod
    def get_public_key_from_node_id(self, node_id: int) -> bytes:
        """Get client's public key in urlsafe bytes for `node_id`."""

    @abc.abstractmethod
    def store_server_public_private_key(
        self, public_key: bytes, private_key: bytes
    ) -> None:
        """Store server's `public_key` and `private_key` in state."""

    @abc.abstractmethod
    def get_server_private_key(self) -> bytes:
        """Get server private key in urlsafe bytes."""

    @abc.abstractmethod
    def get_server_public_key(self) -> bytes:
        """Get server public key in urlsafe bytes."""

    @abc.abstractmethod
    def store_client_public_keys(self, public_keys: Set[bytes]) -> None:
        """Store a set of client public keys in state."""

    @abc.abstractmethod
    def store_client_public_key(self, public_key: bytes) -> None:
        """Retrieve a client public key in state."""

    @abc.abstractmethod
    def get_client_public_keys(self) -> Set[bytes]:
        """Retrieve all currently stored client public keys as a set."""
