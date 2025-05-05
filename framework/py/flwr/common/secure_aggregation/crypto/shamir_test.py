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
"""Tests for Shamir's secret sharing."""


import os
import unittest
from itertools import combinations

from .shamir import combine_shares, create_shares


class TestShamirSecretSharing(unittest.TestCase):
    """Test Shamir's secret sharing implementation."""

    def test_secret_roundtrip(self) -> None:
        """Test that the secret can be reconstructed from shares."""
        # Prepare
        secret = b"My top secret data"
        threshold = 3
        num_shares = 5

        # Execute
        shares = create_shares(secret, threshold, num_shares)

        # Assert
        self.assertEqual(len(shares), num_shares)
        self.assertTrue(all(isinstance(s, bytes) for s in shares))

        # Test all combinations that meet the threshold
        for subset in combinations(shares, threshold):
            reconstructed = combine_shares(list(subset))
            self.assertEqual(reconstructed, secret)

    def test_empty_secret(self) -> None:
        """Test that an empty secret can be reconstructed."""
        # Prepare
        secret = b""
        threshold = 2
        num_shares = 3

        # Execute
        shares = create_shares(secret, threshold, num_shares)
        reconstructed = combine_shares(shares[:threshold])

        # Assert
        self.assertEqual(reconstructed, secret)

    def test_secret_exactly_multiple_of_blocksize(self) -> None:
        """Test that a secret that is exactly a multiple of the block size can be
        reconstructed."""
        # Prepare
        secret = os.urandom(32)  # 32 bytes (2 AES blocks)

        # Execute
        shares = create_shares(secret, 2, 4)
        reconstructed = combine_shares(shares[:2])

        # Assert
        self.assertEqual(reconstructed, secret)

    def test_secret_non_block_aligned(self) -> None:
        """Test that a non-block-aligned secret can be reconstructed."""
        # Prepare
        secret = os.urandom(31)  # Not a multiple of 16

        # Execute
        shares = create_shares(secret, 2, 4)
        reconstructed = combine_shares(shares[:2])

        # Assert
        self.assertEqual(reconstructed, secret)

    def test_invalid_shares_fails(self) -> None:
        """Test that an invalid number of shares fails."""
        # Prepare
        secret = b"Invalid test"
        shares = create_shares(secret, 3, 5)
        bad_shares = shares[:2]  # Not enough for threshold

        try:
            reconstructed = combine_shares(bad_shares)
        except ValueError:
            reconstructed = None

        # Assert
        self.assertNotEqual(reconstructed, secret)

    def test_tampered_share_fails(self) -> None:
        """Test that a tampered share fails to reconstruct the secret."""
        # Prepare
        secret = b"Don't tamper!"
        shares = create_shares(secret, 3, 5)

        # Corrupt the first byte of the first share (not the index bytes)
        corrupted_share = bytearray(shares[0])
        corrupted_share[4] ^= 0xFF
        corrupted_shares = [bytes(corrupted_share)] + shares[1:3]

        # Assert
        self.assertNotEqual(combine_shares(corrupted_shares), secret)
