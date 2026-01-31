from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, ed448
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

from . import eccfrog


@dataclass(frozen=True)
class KoblitzCurve:
    name: str
    key_size_bits: int
    curve: object

    @property
    def key_size_bytes(self) -> int:
        return (self.key_size_bits + 7) // 8


SUPPORTED_CURVES: Dict[str, KoblitzCurve] = {
    "KOBLITZ_112": KoblitzCurve("KOBLITZ_163", 163, ec.SECT163K1()), ##la libreria non supporta 112
    "KOBLITZ_256": KoblitzCurve("KOBLITZ_256", 256, ec.SECP256K1()),
    "KOBLITZ_512": KoblitzCurve("KOBLITZ_512", 571, ec.SECT571K1()),
    "ED25519": KoblitzCurve("ED25519", 256, ed25519.Ed25519PrivateKey),
    "ED448": KoblitzCurve("ED448", 448, ed448.Ed448PrivateKey), ##EdDSA su Curve448.
    "ECCFROG522PP": KoblitzCurve("ECCFROG522PP", 521, "ECCFROG522PP"),
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


def _load_public_key(key: object, curve: KoblitzCurve) -> object:
    if key is None:
        raise ValueError("Chiave pubblica mancante per la curva scelta")
    if curve.name == "ECCFROG522PP":
        return eccfrog.load_public_key(key)
    if isinstance(key, (bytes, bytearray)):
        return load_pem_public_key(bytes(key))
    if isinstance(key, str):
        return load_pem_public_key(key.encode())
    return key


def _load_private_key(key: object, curve: KoblitzCurve) -> object:
    if key is None:
        raise ValueError("Chiave privata mancante per la curva scelta")
    if curve.name == "ECCFROG522PP":
        return eccfrog.load_private_key(key)
    if isinstance(key, (bytes, bytearray)):
        return load_pem_private_key(bytes(key), password=None)
    if isinstance(key, str):
        return load_pem_private_key(key.encode(), password=None)
    return key


def _pack_signature(
    data: bytes, signature: bytes, public_key_bytes: bytes | None = None
) -> bytes:
    if public_key_bytes is None:
        return data + signature + struct.pack(">H", len(signature))
    return (
        data
        + b"FLPK"
        + struct.pack(">H", len(public_key_bytes))
        + public_key_bytes
        + signature
        + struct.pack(">H", len(signature))
    )


def _unpack_signature(payload: bytes) -> tuple[bytes, bytes, bytes | None]:
    if len(payload) < 2:
        raise ValueError("Payload troppo corto per la firma")
    sig_len = struct.unpack(">H", payload[-2:])[0]
    if len(payload) < 2 + sig_len:
        raise ValueError("Payload troppo corto per la firma indicata")
    signature = payload[-2 - sig_len : -2]
    remaining = payload[: -2 - sig_len]
    if len(remaining) >= 6:
        marker_start = remaining.rfind(b"FLPK")
        if marker_start != -1 and len(remaining) >= marker_start + 6:
            public_key_len = struct.unpack(
                ">H", remaining[marker_start + 4 : marker_start + 6]
            )[0]
            public_key_end = marker_start + 6 + public_key_len
            if public_key_end == len(remaining):
                public_key = remaining[marker_start + 6 : public_key_end]
                data = remaining[:marker_start]
                return data, signature, public_key
    data = remaining
    return data, signature, None


def authenticate(data: bytes, curve_name: str, ecc_privkey: object) -> bytes:
    curve = _get_curve(curve_name)
    include_public_key = ecc_privkey is None
    if curve.name == "ECCFROG522PP":
        if ecc_privkey is None:
            private_key = eccfrog.generate_private_key()
        else:
            private_key = _load_private_key(ecc_privkey, curve)
        signature = eccfrog.sign(data, private_key)
        public_key_bytes = (
            private_key.public_key.to_bytes() if include_public_key else None
        )
        return _pack_signature(data, signature, public_key_bytes)
    if ecc_privkey is None:
        if curve.curve is ed25519.Ed25519PrivateKey:
            private_key = ed25519.Ed25519PrivateKey.generate()
        elif curve.curve is ed448.Ed448PrivateKey:
            private_key = ed448.Ed448PrivateKey.generate()
        else:
            private_key = ec.generate_private_key(curve.curve)
    else:
        private_key = _load_private_key(ecc_privkey, curve)
    if isinstance(private_key, (ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey)):
        signature = private_key.sign(data)
        public_key_bytes = (
            private_key.public_key().public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
            )
            if include_public_key
            else None
        )
        return _pack_signature(data, signature, public_key_bytes)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError("Curva non supportata per firme")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    public_key_bytes = (
        private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        if include_public_key
        else None
    )
    return _pack_signature(data, signature, public_key_bytes)


def verify(authenticated_data: bytes, curve_name: str, ecc_pubkey: object) -> bytes:
    curve = _get_curve(curve_name)
    data, signature, embedded_public_key = _unpack_signature(authenticated_data)
    if curve.name == "ECCFROG522PP":
        if ecc_pubkey is None:
            if embedded_public_key is None:
                raise ValueError("Chiave pubblica mancante per la curva scelta")
            public_key = _load_public_key(embedded_public_key, curve)
        else:
            public_key = _load_public_key(ecc_pubkey, curve)
        if not eccfrog.verify(data, signature, public_key):
            raise ValueError("Firma non valida")
        return data
    if ecc_pubkey is None:
        if embedded_public_key is None:
            raise ValueError("Chiave pubblica mancante per la curva scelta")
        public_key = _load_public_key(embedded_public_key, curve)
    else:
        public_key = _load_public_key(ecc_pubkey, curve)
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
