# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ed25519-only asymmetric cryptography utilities."""

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typing import Tuple


def generate_key_pair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate an Ed25519 private/public key pair.

    Returns
    -------
    Tuple[Ed25519PrivateKey, Ed25519PublicKey]
        Private and public key pair.
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def private_key_to_bytes(private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Serialize an Ed25519 private key to PEM bytes.

    Parameters
    ----------
    private_key : Ed25519PrivateKey
        The private key to serialize.

    Returns
    -------
    bytes
        PEM-encoded private key.
    """
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def bytes_to_private_key(private_key_bytes: bytes) -> ed25519.Ed25519PrivateKey:
    """Deserialize an Ed25519 private key from PEM bytes.

    Parameters
    ----------
    private_key_bytes : bytes
        PEM-encoded private key.

    Returns
    -------
    Ed25519PrivateKey
        Deserialized private key.
    """
    return serialization.load_pem_private_key(private_key_bytes, password=None)


def public_key_to_bytes(public_key: ed25519.Ed25519PublicKey) -> bytes:
    """Serialize an Ed25519 public key to PEM bytes.

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The public key to serialize.

    Returns
    -------
    bytes
        PEM-encoded public key.
    """
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def bytes_to_public_key(public_key_bytes: bytes) -> ed25519.Ed25519PublicKey:
    """Deserialize an Ed25519 public key from PEM bytes.

    Parameters
    ----------
    public_key_bytes : bytes
        PEM-encoded public key.

    Returns
    -------
    Ed25519PublicKey
        Deserialized public key.
    """
    return serialization.load_pem_public_key(public_key_bytes)


def sign_message(private_key: ed25519.Ed25519PrivateKey, message: bytes) -> bytes:
    """Sign a message using an Ed25519 private key.

    Parameters
    ----------
    private_key : Ed25519PrivateKey
        The private key used for signing.
    message : bytes
        The message to sign.

    Returns
    -------
    bytes
        The signature of the message.
    """
    return private_key.sign(message)


def verify_signature(public_key: ed25519.Ed25519PublicKey, message: bytes, signature: bytes) -> bool:
    """Verify a signature using an Ed25519 public key.

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The public key used for verification.
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
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False

