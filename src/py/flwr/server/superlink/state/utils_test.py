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
"""Utils tests."""

import unittest

from .utils import generate_rand_int_from_bytes, sint64_to_uint64, uint64_to_sint64


class UtilsTest(unittest.TestCase):
    """Test utils code."""

    def test_uint64_to_sint64(self) -> None:
        """Test conversion from uint64 to sint64."""
        # Test values below 2^63
        self.assertEqual(uint64_to_sint64(0), 0)
        self.assertEqual(uint64_to_sint64(2**62), 2**62)
        self.assertEqual(uint64_to_sint64(2**63 - 1), 2**63 - 1)

        # Test values at and above 2^63
        self.assertEqual(uint64_to_sint64(2**63), -(2**63))
        self.assertEqual(uint64_to_sint64(2**63 + 1), -(2**63) + 1)
        self.assertEqual(uint64_to_sint64(2**64 - 1), -1)

    def test_sint64_to_uint64(self) -> None:
        """Test conversion from sint64 to uint64."""
        # Test values within the range of sint64
        self.assertEqual(sint64_to_uint64(-(2**63)), 2**63)
        self.assertEqual(sint64_to_uint64(-(2**63) + 1), 2**63 + 1)
        self.assertEqual(sint64_to_uint64(-1), 2**64 - 1)
        self.assertEqual(sint64_to_uint64(0), 0)
        self.assertEqual(sint64_to_uint64(2**63 - 1), 2**63 - 1)

        # Test values above 2^63
        self.assertEqual(sint64_to_uint64(2**63), 2**63)
        self.assertEqual(sint64_to_uint64(2**64 - 1), 2**64 - 1)

    def test_generate_rand_int_from_bytes_unsigned_int(self) -> None:
        """Test that the generated integer is unsigned (non-negative)."""
        for num_bytes in range(1, 9):
            with self.subTest(num_bytes=num_bytes):
                rand_int = generate_rand_int_from_bytes(num_bytes)
                self.assertGreaterEqual(rand_int, 0)
