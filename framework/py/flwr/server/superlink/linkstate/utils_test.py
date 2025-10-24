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
    convert_sint64_values_in_dict_to_uint64,
    convert_uint64_values_in_dict_to_sint64,
    generate_rand_int_from_bytes,
)


class UtilsTest(unittest.TestCase):
    """Test utils code."""

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
