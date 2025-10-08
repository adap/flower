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
"""Tests for asymmetric ed25519 specific cryptography utilities."""


from .asymmetric_ed25519 import generate_key_pairs, sign_message, verify_signature


def test_sign_and_verify_success() -> None:
    """Test signing and verifying a message successfully."""
    # Prepare
    private_key, public_key = generate_key_pairs()
    message = b"Test message"

    # Execute
    signature = sign_message(private_key, message)

    # Assert
    assert verify_signature(public_key, message, signature)


def test_sign_and_verify_failure() -> None:
    """Test signing and verifying a message with incorrect keys or data."""
    # Prepare
    private_key, public_key = generate_key_pairs()
    another_public_key = generate_key_pairs()[1]
    message = b"Test message"

    # Execute
    signature = sign_message(private_key, message)

    # Assert
    assert not verify_signature(another_public_key, message, signature)
    assert not verify_signature(public_key, b"Another message", signature)
