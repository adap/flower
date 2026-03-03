"""fedhomo: Homomorphic encryption utilities using TenSEAL (CKKS scheme)."""

import logging
from functools import reduce
from typing import List, Tuple

import numpy as np
import tenseal as ts

log = logging.getLogger(__name__)

PUBLIC_CONTEXT_PATH = "keys/public_context.pkl"
SECRET_CONTEXT_PATH = "keys/secret_context.pkl"


class HomomorphicError(Exception):
    """Base exception for homomorphic encryption operations."""


class EncryptionError(HomomorphicError):
    """Raised when encryption fails."""


class DecryptionError(HomomorphicError):
    """Raised when decryption fails."""


class EncryptedAggregator:
    """Handles secure aggregation of encrypted model updates on the server."""

    def __init__(self, context: ts.Context):
        if not context.is_public():
            raise ValueError("Aggregator requires a public context.")
        self.context = context

    def process_client_update(self, encrypted_params: List[np.ndarray]) -> List[ts.CKKSVector]:
        """Deserialize raw bytes from a client into CKKSVectors."""
        vectors = []
        for param in encrypted_params:
            vec = ts.ckks_vector_from(self.context, param.tobytes())
            vec.link_context(self.context)
            vectors.append(vec)
        return vectors

    def weighted_sum(
        self, updates: List[Tuple[List[ts.CKKSVector], int]]
    ) -> List[ts.CKKSVector]:
        """Compute weighted average of encrypted parameter updates.

        Args:
            updates: List of (vectors, num_examples) tuples from each client.

        Returns:
            List of aggregated CKKSVectors.
        """
        total_weight = sum(weight for _, weight in updates)
        weighted = [
            [v * (weight / total_weight) for v in vectors]
            for vectors, weight in updates
        ]
        return [reduce(lambda a, b: a + b, layer) for layer in zip(*weighted)]

    def serialize_vectors(self, vectors: List[ts.CKKSVector]) -> List[bytes]:
        """Serialize aggregated CKKSVectors for transmission to clients."""
        return [v.serialize() for v in vectors]


class HomomorphicClientHandler:
    """Handles CKKS encryption and decryption of model parameters on the client."""

    def __init__(self, cid: str):
        self.cid = cid
        self.logger = logging.getLogger(f"CryptoHandler-{cid}")
        self.public_context, self.secret_context = self._load_contexts()
        self._validate_contexts()

    def _load_contexts(self) -> Tuple[ts.Context, ts.Context]:
        """Load public and secret TenSEAL contexts from disk."""
        try:
            with open(PUBLIC_CONTEXT_PATH, "rb") as f:
                public_ctx = ts.context_from(f.read())
            with open(SECRET_CONTEXT_PATH, "rb") as f:
                secret_ctx = ts.context_from(f.read())
            return public_ctx, secret_ctx
        except Exception as e:
            self.logger.error("Context loading failed: %s", e)
            raise EncryptionError("Security context initialization failed") from e

    def _validate_contexts(self) -> None:
        """Validate that loaded contexts have the correct visibility."""
        if not self.public_context.is_public():
            raise EncryptionError("Public context must be public.")
        if self.secret_context.is_public():
            raise EncryptionError("Secret context must not be public.")

    def encrypt_parameters(self, ndarrays: List[np.ndarray]) -> List[bytes]:
        """Encrypt model parameters using CKKS.

        Each array is flattened to 1D before encryption.

        Args:
            ndarrays: List of model parameter arrays.

        Returns:
            List of serialized encrypted vectors.
        """
        try:
            return [
                ts.ckks_vector(self.public_context, arr.flatten().tolist()).serialize()
                for arr in ndarrays
            ]
        except Exception as e:
            self.logger.error("Encryption failed: %s", e)
            raise EncryptionError("Parameter encryption failed") from e

    def decrypt_parameters(self, encrypted_data: List[bytes]) -> List[np.ndarray]:
        """Decrypt received CKKS-encrypted parameters.

        Args:
            encrypted_data: List of serialized CKKSVectors.

        Returns:
            List of decrypted numpy arrays (1D, shapes not yet restored).
        """
        try:
            result = []
            for data in encrypted_data:
                vec = ts.ckks_vector_from(self.secret_context, data)
                vec.link_context(self.secret_context)
                result.append(np.array(vec.decrypt()))
            return result
        except Exception as e:
            self.logger.error("Decryption failed: %s", e)
            raise DecryptionError("Vector decryption failed") from e

    def process_incoming_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Deserialize and decrypt incoming parameters from the server.

        Falls back gracefully on round 1 when parameters are plaintext.

        Args:
            parameters: List of ndarrays wrapping serialized CKKS bytes.

        Returns:
            List of decrypted 1D numpy arrays.
        """
        if not parameters:
            raise DecryptionError("Empty parameters received.")
        try:
            return self.decrypt_parameters([p.tobytes() for p in parameters])
        except Exception as e:
            self.logger.debug(
                "Could not decrypt parameters (expected on round 1 — plaintext fallback): %s", e
            )
            raise DecryptionError("Parameter deserialization failed") from e
