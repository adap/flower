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
"""Tests for utility functions for the infrastructure."""


from parameterized import parameterized

from .utils import int64_to_uint64, mask_string, uint64_to_int64


def test_mask_string() -> None:
    """Test the `mask_string` function."""
    assert mask_string("abcdefghi") == "abcd...fghi"
    assert mask_string("abcdefghijklm") == "abcd...jklm"
    assert mask_string("abc") == "abc"
    assert mask_string("a") == "a"
    assert mask_string("") == ""
    assert mask_string("1234567890", head=2, tail=3) == "12...890"
    assert mask_string("1234567890", head=5, tail=4) == "12345...7890"


@parameterized.expand(  # type: ignore
    [
        # Test values within the positive range of sint64 (below 2^63)
        (0, 0),  # Minimum positive value
        (1, 1),  # 1 remains 1 in both uint64 and sint64
        (2**62, 2**62),  # Mid-range positive value
        (2**63 - 1, 2**63 - 1),  # Maximum positive value for sint64
        # Test values at or above 2^63 (become negative in sint64)
        (2**63, -(2**63)),  # Minimum negative value for sint64
        (2**63 + 1, -(2**63) + 1),  # Slightly above the boundary
        (9223372036854775811, -9223372036854775805),  # Some value > sint64 max
        (2**64 - 1, -1),  # Maximum uint64 value becomes -1 in sint64
    ]
)
def test_convert_uint64_to_sint64(before: int, after: int) -> None:
    """Test conversion from uint64 to sint64."""
    assert uint64_to_int64(before) == after


@parameterized.expand(  # type: ignore
    [
        # Test values within the negative range of sint64
        (-(2**63), 2**63),  # Minimum sint64 value becomes 2^63 in uint64
        (-(2**63) + 1, 2**63 + 1),  # Slightly above the minimum
        (-9223372036854775805, 9223372036854775811),  # Some value > sint64 max
        # Test zero-adjacent inputs
        (-1, 2**64 - 1),  # -1 in sint64 becomes 2^64 - 1 in uint64
        (0, 0),  # 0 remains 0 in both sint64 and uint64
        (1, 1),  # 1 remains 1 in both sint64 and uint64
        # Test values within the positive range of sint64
        (2**63 - 1, 2**63 - 1),  # Maximum positive value in sint64
        # Test boundary and maximum uint64 value
        (2**63, 2**63),  # Exact boundary value for sint64
        (2**64 - 1, 2**64 - 1),  # Maximum uint64 value, stays the same
    ]
)
def test_sint64_to_uint64(before: int, after: int) -> None:
    """Test conversion from sint64 to uint64."""
    assert int64_to_uint64(before) == after


@parameterized.expand(  # type: ignore
    [
        (0),
        (1),
        (2**62),
        (2**63 - 1),
        (2**63),
        (2**63 + 1),
        (9223372036854775811),
        (2**64 - 1),
    ]
)
def test_uint64_to_sint64_to_uint64(expected: int) -> None:
    """Test conversion from sint64 to uint64."""
    actual = int64_to_uint64(uint64_to_int64(expected))
    assert actual == expected
