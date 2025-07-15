"""Test Compressors."""

import unittest

import numpy as np

from dasha.compressors import IdentityUnbiasedCompressor, RandKCompressor, decompress


class TestIdentityUnbiasedCompressor(unittest.TestCase):
    """Test."""

    def test(self) -> None:
        """Test."""
        vec = np.random.rand(10)
        compressed_vec = IdentityUnbiasedCompressor().compress(vec)
        np.testing.assert_almost_equal(
            vec,
            decompress(
                compressed_vec, assert_compressor=IdentityUnbiasedCompressor.name()
            ),
        )


class TestUnbiasedBaseCompressor(unittest.TestCase):
    """Test."""

    def test(self) -> None:
        """Test."""
        seed = 42
        number_of_coordinates = 5
        dim = 19
        compressor = RandKCompressor(seed, number_of_coordinates, dim)
        vector = np.random.randn(dim).astype(np.float32)
        number_of_samples = 100000
        expected_vector = 0
        for _ in range(number_of_samples):
            compressed_vector = compressor.compress(vector)
            decompressed_vector = decompress(compressed_vector)
            assert (
                np.count_nonzero(decompressed_vector)
                == compressor.num_nonzero_components()
            )
            expected_vector += decompressed_vector / number_of_samples
        np.testing.assert_array_almost_equal(expected_vector, vector, decimal=1)


if __name__ == "__main__":
    unittest.main()
