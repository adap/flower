"""Defines compressors for the Dasha and Marina methods."""

import numpy as np


def decompress(compressed_vector, assert_compressor=None):
    """Decompress a compressed vector."""
    dim, indices, values, name = compressed_vector
    if assert_compressor is not None:
        assert assert_compressor == name
    decompressed_array = np.zeros((dim,), dtype=values.dtype)
    decompressed_array[indices] = values
    return decompressed_array


BITS_IN_FLOAT_32 = 32


def estimate_size(compressed_vector):
    """Estimate size of a compressed vector in bits."""
    _, indices, values, _ = compressed_vector
    assert values.dtype == np.float32
    return (len(indices) + len(values)) * BITS_IN_FLOAT_32


class BaseCompressor:
    """Abstract compressor class."""

    def __init__(self, seed, dim) -> None:
        self._seed = seed
        self._dim = dim

    def compress(self, vector):
        """Compress vector."""
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        compressed_vector = self.compress_impl(vector)
        class_name = self.name()
        compressed_vector.append(class_name)
        return compressed_vector

    @classmethod
    def name(cls):
        """Return a compressor's name."""
        return cls.__name__

    def set_dim(self, dim):
        """Set the dim of input vectors."""
        assert self._dim is None or self._dim == dim
        self._dim = dim

    def compress_impl(self, vector):
        """Implement the compressor's logic."""
        raise NotImplementedError()

    def num_nonzero_components(self):
        """Return the number of coordinates that are preserved."""
        raise NotImplementedError()


class UnbiasedBaseCompressor(BaseCompressor):
    """Abstract unbiased compressor class."""

    def omega(self):
        """Return the variance of the compressor."""
        raise NotImplementedError()


class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    """Identity unbiased compressor that does not compress."""

    def __init__(self, seed=None, dim=None):
        super().__init__(seed=seed, dim=dim)

    def compress_impl(self, vector):
        """Implement the compressor's logic."""
        dim = self._dim
        return [np.array(dim, dtype=np.int32), np.arange(dim), np.copy(vector)]

    def omega(self):
        """Return the variance of the compressor."""
        return 0

    def num_nonzero_components(self):
        """Return the number of coordinates that are preserved."""
        assert self._dim is not None
        return self._dim


class RandKCompressor(UnbiasedBaseCompressor):
    """RandomK unbiased compressor.

    Takes random number_of_coordinates coordinates in a vector.
    """

    def __init__(self, seed, number_of_coordinates, dim=None):
        super().__init__(seed=seed, dim=dim)
        self._number_of_coordinates = number_of_coordinates
        self._generator = np.random.default_rng(seed)

    def num_nonzero_components(self):
        """Return the number of coordinates that are preserved."""
        return self._number_of_coordinates

    def compress_impl(self, vector):
        """Implement the compressor's logic."""
        assert self._number_of_coordinates >= 0
        dim = self._dim
        indices = self._generator.choice(
            dim, self._number_of_coordinates, replace=False
        )
        values = vector[indices] * float(dim / self._number_of_coordinates)
        return [np.array(dim, dtype=np.int32), indices, values]

    def omega(self):
        """Return the variance of the compressor."""
        assert self._dim is not None
        return float(self._dim) / self._number_of_coordinates - 1
