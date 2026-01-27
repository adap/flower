"""Autenticazione reale con curve ellittiche (ECDSA)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, ed448
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)


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
    "ECDSA_256": KoblitzCurve("ECDSA_256", 256, ec.SECP256R1()),
    "ECDSA_521": KoblitzCurve("ECDSA_521", 521, ec.SECP521R1()),
    "KOBLITZ_112": KoblitzCurve("KOBLITZ_112", 163, ec.SECT163K1()),
    "KOBLITZ_256": KoblitzCurve("KOBLITZ_256", 256, ec.SECP256K1()),
    "KOBLITZ_512": KoblitzCurve("KOBLITZ_512", 571, ec.SECT571K1()),
    "CURVE25519": KoblitzCurve("CURVE25519", 256, ed25519.Ed25519PrivateKey),
    "CURVE448": KoblitzCurve("CURVE448", 448, ed448.Ed448PrivateKey),
    "ECCFROG522PP": KoblitzCurve("ECCFROG522PP", 521, ec.SECP521R1()),
}

LEGACY_ALIASES: Dict[str, str] = {}

SUPPORTED_METHODS = set(SUPPORTED_CURVES.keys()) | set(LEGACY_ALIASES.keys())


def _get_curve(curve_name: str) -> KoblitzCurve:
    normalized_name = LEGACY_ALIASES.get(curve_name, curve_name)
    try:
        return SUPPORTED_CURVES[normalized_name]
    except KeyError as exc:  # pragma: no cover - safety net
        raise ValueError(f"Curva Koblitz non supportata: {curve_name}") from exc


def is_supported_method(curve_name: str) -> bool:
    return curve_name in SUPPORTED_METHODS


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


def authenticate(data: bytes, curve_name: str, ecc_privkey: object) -> bytes:
    """Autentica i dati usando firme reali."""

    curve = _get_curve(curve_name)
    private_key = _load_private_key(ecc_privkey)
    if isinstance(private_key, (ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey)):
        signature = private_key.sign(data)
        return _pack_signature(data, signature)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError("Curva non supportata per firme")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return _pack_signature(data, signature)


def verify(authenticated_data: bytes, curve_name: str, ecc_pubkey: object) -> bytes:
    """Verifica l'autenticazione creata da :func:`authenticate`."""

    curve = _get_curve(curve_name)
    public_key = _load_public_key(ecc_pubkey)
    data, signature = _unpack_signature(authenticated_data)
    try:
        if isinstance(public_key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
            public_key.verify(signature, data)
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        else:
            raise ValueError("Curva non supportata per firme")
    except InvalidSignature as exc:
        raise ValueError("Firma non valida") from exc
    return data
