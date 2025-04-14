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
"""Tests for gRPC."""


from .grpc import valid_certificates


def test_valid_certificates_when_correct() -> None:
    """Test is validation function works correctly when passed valid list."""
    # Prepare
    certificates = (b"a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)

    # Assert
    assert is_valid


def test_valid_certificates_when_wrong() -> None:
    """Test is validation function works correctly when passed invalid list."""
    # Prepare
    certificates = ("not_a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)  # type: ignore

    # Assert
    assert not is_valid
