"""Implementazioni con curve ellittiche reali per la demo."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.asymmetric import ec, x25519, x448
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


@dataclass(frozen=True)
class KoblitzCurve:
    """Definizione di una curva di Koblitz gestita dall'applicazione."""

    name: str
    key_size_bits: int
    curve: object

    @property
    def key_size_bytes(self) -> int:
        return (self.key_size_bits + 7) // 8


SUPPORTED_CURVES: Dict[str, KoblitzCurve] = {
    "KOBLITZ_SMALL": KoblitzCurve("KOBLITZ_SMALL", 192, ec.SECP192R1()),
    "KOBLITZ_MEDIUM": KoblitzCurve("KOBLITZ_MEDIUM", 256, ec.SECP256R1()),
    "KOBLITZ_LARGE": KoblitzCurve("KOBLITZ_LARGE", 521, ec.SECP521R1()),
    "CURVE25519": KoblitzCurve("CURVE25519", 256, x25519),
    "CURVE448": KoblitzCurve("CURVE448", 448, x448),
    "ECCFROG522PP": KoblitzCurve("ECCFROG522PP", 521, ec.SECP521R1()),
}

LEGACY_ALIASES: Dict[str, str] = {
    "KOBLITZ_112": "KOBLITZ_SMALL",
    "KOBLITZ_256": "KOBLITZ_MEDIUM",
    "KOBLITZ_512": "KOBLITZ_LARGE",
}

SUPPORTED_METHODS = set(SUPPORTED_CURVES.keys()) | set(LEGACY_ALIASES.keys())


def _get_curve(curve_name: str) -> KoblitzCurve:
    normalized_name = LEGACY_ALIASES.get(curve_name, curve_name)
    try:
        return SUPPORTED_CURVES[normalized_name]
    except KeyError as exc:  # pragma: no cover - safety net
        raise ValueError(f"Curva Koblitz non supportata: {curve_name}") from exc


def is_supported_method(curve_name: str) -> bool:
    return curve_name in SUPPORTED_METHODS


def _derive_keystream(curve: KoblitzCurve, secret: bytes, length: int) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=curve.name.encode(),
    )
    prk = hkdf.derive(secret)
    keystream = bytearray()
    counter = 1
    while len(keystream) < length:
        hmac_ctx = hmac.HMAC(prk, hashes.SHA256())
        hmac_ctx.update(counter.to_bytes(4, "big"))
        keystream.extend(hmac_ctx.finalize())
        counter += 1
    return bytes(keystream[:length])


def _load_public_key(key: object) -> object:
    if key is None:
        raise ValueError("Chiave pubblica mancante per la curva scelta")
    if isinstance(key, (bytes, bytearray)):
        return load_pem_public_key(bytes(key))
    if isinstance(key, str):
        return load_pem_public_key(key.encode())
    return key


def _load_private_key(key: object) -> object:
    if key is None:
        raise ValueError("Chiave privata mancante per la curva scelta")
    if isinstance(key, (bytes, bytearray)):
        return load_pem_private_key(bytes(key), password=None)
    if isinstance(key, str):
        return load_pem_private_key(key.encode(), password=None)
    return key


def _pack_ephemeral(ephemeral_public: bytes, payload: bytes) -> bytes:
    return struct.pack(">H", len(ephemeral_public)) + ephemeral_public + payload


def _unpack_ephemeral(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < 2:
        raise ValueError("Payload troppo corto per contenere la chiave effimera")
    (pub_len,) = struct.unpack(">H", payload[:2])
    if len(payload) < 2 + pub_len:
        raise ValueError("Payload troppo corto per la chiave effimera indicata")
    return payload[2:2 + pub_len], payload[2 + pub_len:]


def _pack_signature(data: bytes, signature: bytes) -> bytes:
    return data + struct.pack(">H", len(signature)) + signature


def _unpack_signature(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < 2:
        raise ValueError("Payload troppo corto per la firma")
    sig_len = struct.unpack(">H", payload[-2:])[0]
    if len(payload) < 2 + sig_len:
        raise ValueError("Payload troppo corto per la firma indicata")
    data = payload[: -2 - sig_len]
    signature = payload[-2 - sig_len : -2]
    return data, signature


def _generate_ephemeral(curve: KoblitzCurve) -> tuple[object, bytes]:
    if curve.curve is x25519:
        private_key = x25519.X25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return private_key, public_key
    if curve.curve is x448:
        private_key = x448.X448PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return private_key, public_key
    private_key = ec.generate_private_key(curve.curve)
    public_key = private_key.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    return private_key, public_key


def _derive_shared_secret(curve: KoblitzCurve, private_key: object, public_key: object) -> bytes:
    if curve.curve is x25519:
        return private_key.exchange(public_key)
    if curve.curve is x448:
        return private_key.exchange(public_key)
    return private_key.exchange(ec.ECDH(), public_key)


def _load_ephemeral_public(curve: KoblitzCurve, public_bytes: bytes) -> object:
    if curve.curve is x25519:
        return x25519.X25519PublicKey.from_public_bytes(public_bytes)
    if curve.curve is x448:
        return x448.X448PublicKey.from_public_bytes(public_bytes)
    return ec.EllipticCurvePublicKey.from_encoded_point(curve.curve, public_bytes)


def encrypt(data: bytes, curve_name: str, ecc_pubkey: object) -> bytes:
    """Cifra i dati utilizzando una curva ellittica reale (ECDH + XOR)."""

    curve = _get_curve(curve_name)
    public_key = _load_public_key(ecc_pubkey)
    ephemeral_private, ephemeral_public = _generate_ephemeral(curve)
    shared_secret = _derive_shared_secret(curve, ephemeral_private, public_key)
    keystream = _derive_keystream(curve, shared_secret, len(data))
    ciphertext = bytes(d ^ k for d, k in zip(data, keystream))
    return _pack_ephemeral(ephemeral_public, ciphertext)


def decrypt(encrypted_data: bytes, curve_name: str, ecc_privkey: object) -> bytes:
    """Decifra i dati protetti con :func:`encrypt`."""

    curve = _get_curve(curve_name)
    private_key = _load_private_key(ecc_privkey)
    ephemeral_public_bytes, ciphertext = _unpack_ephemeral(encrypted_data)
    ephemeral_public = _load_ephemeral_public(curve, ephemeral_public_bytes)
    shared_secret = _derive_shared_secret(curve, private_key, ephemeral_public)
    keystream = _derive_keystream(curve, shared_secret, len(ciphertext))
    return bytes(c ^ k for c, k in zip(ciphertext, keystream))


def authenticate(data: bytes, curve_name: str, ecc_privkey: object) -> bytes:
    """Autentica i dati usando firme reali."""

    curve = _get_curve(curve_name)
    private_key = _load_private_key(ecc_privkey)
    if curve.curve in (x25519, x448):
        raise ValueError("Le curve X25519/X448 non supportano la firma diretta")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return _pack_ephemeral(b"", _pack_signature(data, signature))


def verify(authenticated_data: bytes, curve_name: str, ecc_pubkey: object) -> bytes:
    """Verifica l'autenticazione creata da :func:`authenticate`."""

    curve = _get_curve(curve_name)
    public_key = _load_public_key(ecc_pubkey)
    _, payload = _unpack_ephemeral(authenticated_data)
    if curve.curve in (x25519, x448):
        raise ValueError("Le curve X25519/X448 non supportano la firma diretta")
    data, signature = _unpack_signature(payload)
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise ValueError("Autenticazione ellittica non valida") from exc
    return data
