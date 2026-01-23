"""Implementazioni semplificate di curve di Koblitz per la demo.

Questo modulo fornisce solo generazione chiavi e autenticazione (firma/verifica).
Non espone funzioni di cifratura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


@dataclass(frozen=True)
class KoblitzCurve:
    """Definizione di una curva di Koblitz gestita dall'applicazione."""

    name: str
    curve: ec.EllipticCurve

    @property
    def key_size_bytes(self) -> int:
        return (self.curve.key_size + 7) // 8


SUPPORTED_CURVES: Dict[str, KoblitzCurve] = {
    # Le dimensioni originali (112/256/512) sono mappate a curve supportate
    # da cryptography per la demo.
    "KOBLITZ_SMALL": KoblitzCurve("KOBLITZ_SMALL", ec.SECP192R1()),
    "KOBLITZ_MEDIUM": KoblitzCurve("KOBLITZ_MEDIUM", ec.SECP256K1()),
    "KOBLITZ_LARGE": KoblitzCurve("KOBLITZ_LARGE", ec.SECP521R1()),
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


def _ensure_curve_matches(curve: KoblitzCurve, key_curve: ec.EllipticCurve) -> None:
    if key_curve.name != curve.curve.name:
        raise ValueError(
            "La chiave non corrisponde alla curva selezionata: "
            f"attesa {curve.curve.name}, trovata {key_curve.name}"
        )


def generate_keypair(curve_name: str) -> tuple[bytes, bytes]:
    """Genera una coppia di chiavi (privata, pubblica) in formato PEM."""

    curve = _get_curve(curve_name)
    private_key = ec.generate_private_key(curve.curve)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def sign(data: bytes, private_key_pem: bytes, curve_name: str) -> bytes:
    """Firma i dati con la chiave privata usando ECDSA."""
    curve = _get_curve(curve_name)
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise TypeError("Chiave privata non valida per ECDSA")
    _ensure_curve_matches(curve, private_key.curve)
    return private_key.sign(data, ec.ECDSA(hashes.SHA256()))


def verify(data: bytes, signature: bytes, public_key_pem: bytes, curve_name: str) -> bool:
    """Verifica la firma ECDSA usando la chiave pubblica."""
    curve = _get_curve(curve_name)
    public_key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise TypeError("Chiave pubblica non valida per ECDSA")
    _ensure_curve_matches(curve, public_key.curve)
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False
    return True
