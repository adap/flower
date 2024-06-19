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
"""Symmetric encryption tests."""


from .symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    verify_hmac,
)


def test_generate_shared_key() -> None:
    """Test util function generate_shared_key."""
    # Prepare
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()

    # Execute
    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])

    # Assert
    assert client_shared_secret == server_shared_secret


def test_wrong_secret_generate_shared_key() -> None:
    """Test util function generate_shared_key with wrong secret."""
    # Prepare
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    other_keys = generate_key_pairs()

    # Execute
    client_shared_secret = generate_shared_key(client_keys[0], other_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])

    # Assert
    assert client_shared_secret != server_shared_secret


def test_hmac() -> None:
    """Test util function compute and verify hmac."""
    # Prepare
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])
    message = b"Flower is the future of AI"

    # Execute
    client_compute_hmac = compute_hmac(client_shared_secret, message)

    # Assert
    assert verify_hmac(server_shared_secret, message, client_compute_hmac)


def test_wrong_secret_hmac() -> None:
    """Test util function compute and verify hmac with wrong secret."""
    # Prepare
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    other_keys = generate_key_pairs()
    client_shared_secret = generate_shared_key(client_keys[0], other_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])
    message = b"Flower is the future of AI"

    # Execute
    client_compute_hmac = compute_hmac(client_shared_secret, message)

    # Assert
    assert verify_hmac(server_shared_secret, message, client_compute_hmac) is False


def test_wrong_message_hmac() -> None:
    """Test util function compute and verify hmac with wrong message."""
    # Prepare
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])
    message = b"Flower is the future of AI"
    other_message = b"Flower is not the future of AI"

    # Execute
    client_compute_hmac = compute_hmac(client_shared_secret, other_message)

    # Assert
    assert verify_hmac(server_shared_secret, message, client_compute_hmac) is False