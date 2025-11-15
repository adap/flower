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
"""Asymmetric cryptography utilities."""


from typing import cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


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


def uses_nist_ec_curve(public_key: ec.EllipticCurvePublicKey) -> bool:
    """Return True if the provided key uses a NIST EC curve."""
    return isinstance(
        public_key.curve,
        (ec.SECP192R1 | ec.SECP224R1 | ec.SECP256R1 | ec.SECP384R1 | ec.SECP521R1),
    )
