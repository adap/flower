"""Autenticazione reale con curve ellittiche (ECDSA)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)


@dataclass(frozen=True)
class KoblitzCurve:
    """Definizione di una curva di Koblitz gestita dall'applicazione."""

    name: str
    key_size_bits: int
    curve: ec.EllipticCurve

    @property
    def key_size_bytes(self) -> int:
        return (self.key_size_bits + 7) // 8


SUPPORTED_CURVES: Dict[str, KoblitzCurve] = {
    "ECDSA_256": KoblitzCurve("ECDSA_256", 256, ec.SECP256R1()),
    "ECDSA_521": KoblitzCurve("ECDSA_521", 521, ec.SECP521R1()),
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
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError("Curva non supportata per firme")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return _pack_signature(data, signature)


def verify(authenticated_data: bytes, curve_name: str, ecc_pubkey: object) -> bytes:
    """Verifica l'autenticazione creata da :func:`authenticate`."""

    curve = _get_curve(curve_name)
    public_key = _load_public_key(ecc_pubkey)
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise ValueError("Curva non supportata per firme")
    data, signature = _unpack_signature(authenticated_data)
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise ValueError("Firma non valida") from exc
    return data
