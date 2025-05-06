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
"""Symmetric encryption."""


import base64
from typing import cast

from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_key_pairs() -> (
    tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
):
    """Generate private and public key pairs with Cryptography."""
    private_key = ec.generate_private_key(ec.SECP384R1())
    public_key = private_key.public_key()
    return private_key, public_key


def private_key_to_bytes(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """Serialize private key to bytes."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def bytes_to_private_key(private_key_bytes: bytes) -> ec.EllipticCurvePrivateKey:
    """Deserialize private key from bytes."""
    return cast(
        ec.EllipticCurvePrivateKey,
        serialization.load_pem_private_key(data=private_key_bytes, password=None),
    )


def public_key_to_bytes(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """Serialize public key to bytes."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def bytes_to_public_key(public_key_bytes: bytes) -> ec.EllipticCurvePublicKey:
    """Deserialize public key from bytes."""
    return cast(
        ec.EllipticCurvePublicKey,
        serialization.load_pem_public_key(data=public_key_bytes),
    )


def generate_shared_key(
    private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey
) -> bytes:
    """Generate a shared key from a private key (i.e., a secret key) and a public key.

    Generate shared key by exchange function and key derivation function Key derivation
    function is needed to obtain final shared key of exactly 32 bytes
    """
    # Generate a 32 byte urlsafe(for fernet) shared key
    # from own private key and another public key
    shared_key = private_key.exchange(ec.ECDH(), public_key)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
    ).derive(shared_key)
    return base64.urlsafe_b64encode(derived_key)


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt plaintext using 32-byte key with Fernet."""
    # The input key must be url safe
    fernet = Fernet(key)
    return fernet.encrypt(plaintext)


def decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt ciphertext using 32-byte key with Fernet."""
    # The input key must be url safe
    fernet = Fernet(key)
    return fernet.decrypt(ciphertext)


def compute_hmac(key: bytes, message: bytes) -> bytes:
    """Compute hmac of a message using key as hash."""
    computed_hmac = hmac.HMAC(key, hashes.SHA256())
    computed_hmac.update(message)
    return computed_hmac.finalize()


def verify_hmac(key: bytes, message: bytes, hmac_value: bytes) -> bool:
    """Verify hmac of a message using key as hash."""
    computed_hmac = hmac.HMAC(key, hashes.SHA256())
    computed_hmac.update(message)
    try:
        computed_hmac.verify(hmac_value)
        return True
    except InvalidSignature:
        return False


def sign_message(private_key: ec.EllipticCurvePrivateKey, message: bytes) -> bytes:
    """Sign a message using the provided EC private key.

    Parameters
    ----------
    private_key : ec.EllipticCurvePrivateKey
        The EC private key to sign the message with.
    message : bytes
        The message to be signed.

    Returns
    -------
    bytes
        The signature of the message.
    """
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    return signature


def verify_signature(
    public_key: ec.EllipticCurvePublicKey, message: bytes, signature: bytes
) -> bool:
    """Verify a signature against a message using the provided EC public key.

    Parameters
    ----------
    public_key : ec.EllipticCurvePublicKey
        The EC public key to verify the signature.
    message : bytes
        The original message.
    signature : bytes
        The signature to verify.

    Returns
    -------
    bool
        True if the signature is valid, False otherwise.
    """
    try:
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False
