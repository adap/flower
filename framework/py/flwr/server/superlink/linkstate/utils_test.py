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
"""Utils tests."""


import unittest

from parameterized import parameterized

from .utils import (
    convert_sint64_to_uint64,
    convert_sint64_values_in_dict_to_uint64,
    convert_uint64_to_sint64,
    convert_uint64_values_in_dict_to_sint64,
    generate_rand_int_from_bytes,
)


class UtilsTest(unittest.TestCase):
    """Test utils code."""

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
    def test_convert_uint64_to_sint64(self, before: int, after: int) -> None:
        """Test conversion from uint64 to sint64."""
        self.assertEqual(convert_uint64_to_sint64(before), after)

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
    def test_sint64_to_uint64(self, before: int, after: int) -> None:
        """Test conversion from sint64 to uint64."""
        self.assertEqual(convert_sint64_to_uint64(before), after)

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
    def test_uint64_to_sint64_to_uint64(self, expected: int) -> None:
        """Test conversion from sint64 to uint64."""
        actual = convert_sint64_to_uint64(convert_uint64_to_sint64(expected))
        self.assertEqual(expected, actual)

    @parameterized.expand(  # type: ignore
        [
            # Test cases with uint64 values
            (
                {"a": 0, "b": 2**63 - 1, "c": 2**63, "d": 2**64 - 1},
                ["a", "b", "c", "d"],
                {"a": 0, "b": 2**63 - 1, "c": -(2**63), "d": -1},
            ),
            (
                {"a": 1, "b": 2**62, "c": 2**63 + 1},
                ["a", "b", "c"],
                {"a": 1, "b": 2**62, "c": -(2**63) + 1},
            ),
            # Edge cases with mixed uint64 values and keys
            (
                {"a": 2**64 - 1, "b": 12345, "c": 0},
                ["a", "b"],
                {"a": -1, "b": 12345, "c": 0},
            ),
        ]
    )
    def test_convert_uint64_values_in_dict_to_sint64(
        self, input_dict: dict[str, int], keys: list[str], expected_dict: dict[str, int]
    ) -> None:
        """Test uint64 to sint64 conversion in a dictionary."""
        convert_uint64_values_in_dict_to_sint64(input_dict, keys)
        self.assertEqual(input_dict, expected_dict)

    @parameterized.expand(  # type: ignore
        [
            # Test cases with sint64 values
            (
                {"a": 0, "b": 2**63 - 1, "c": -(2**63), "d": -1},
                ["a", "b", "c", "d"],
                {"a": 0, "b": 2**63 - 1, "c": 2**63, "d": 2**64 - 1},
            ),
            (
                {"a": -1, "b": -(2**63) + 1, "c": 12345},
                ["a", "b", "c"],
                {"a": 2**64 - 1, "b": 2**63 + 1, "c": 12345},
            ),
            # Edge cases with mixed sint64 values and keys
            (
                {"a": -1, "b": 12345, "c": 0},
                ["a", "b"],
                {"a": 2**64 - 1, "b": 12345, "c": 0},
            ),
        ]
    )
    def test_convert_sint64_values_in_dict_to_uint64(
        self, input_dict: dict[str, int], keys: list[str], expected_dict: dict[str, int]
    ) -> None:
        """Test sint64 to uint64 conversion in a dictionary."""
        convert_sint64_values_in_dict_to_uint64(input_dict, keys)
        self.assertEqual(input_dict, expected_dict)

    def test_generate_rand_int_from_bytes_unsigned_int(self) -> None:
        """Test that the generated integer is unsigned (non-negative)."""
        for num_bytes in range(1, 9):
            with self.subTest(num_bytes=num_bytes):
                rand_int = generate_rand_int_from_bytes(num_bytes)
                self.assertGreaterEqual(rand_int, 0)
