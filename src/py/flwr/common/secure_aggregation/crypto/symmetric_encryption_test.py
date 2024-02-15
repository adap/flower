# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Symmetric encryption tests."""

from .symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    verify_hmac,
)


def test_generate_shared_key() -> None:
    """Test util function generate_shared_key."""
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()

    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])

    assert client_shared_secret == server_shared_secret


def test_hmac() -> None:
    """Test util function compute and verify hmac."""
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])
    message = b"Flower is the future of AI"

    client_compute_hmac = compute_hmac(client_shared_secret, message)

    assert verify_hmac(server_shared_secret, message, client_compute_hmac)
