"""Implementazioni semplificate di curve di Koblitz per la demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


@dataclass(frozen=True)
class KoblitzCurve:
    """Definizione di una curva di Koblitz gestita dall'applicazione."""

    name: str
    key_size_bits: int

    @property
    def key_size_bytes(self) -> int:
        return (self.key_size_bits + 7) // 8


SUPPORTED_CURVES: Dict[str, KoblitzCurve] = {
    "KOBLITZ_SMALL": KoblitzCurve("KOBLITZ_SMALL", 112),
    "KOBLITZ_MEDIUM": KoblitzCurve("KOBLITZ_MEDIUM", 256),
    "KOBLITZ_LARGE": KoblitzCurve("KOBLITZ_LARGE", 512),
    "CURVE25519": KoblitzCurve("CURVE25519", 256),
    "CURVE448": KoblitzCurve("CURVE448", 448),
    "ECCFROG522PP": KoblitzCurve("ECCFROG522PP", 522),
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


def encrypt(data: bytes, curve_name: str) -> bytes:
    """Cifra i dati utilizzando una curva di Koblitz simulata.

    La funzione genera un segreto effimero della dimensione della curva
    (112/256/512 bit) e lo usa per derivare un keystream tramite HKDF. Il
    keystream viene poi combinato con i dati tramite XOR. Il segreto viene
    prefissato al ciphertext per consentire la decifratura.
    """

    curve = _get_curve(curve_name)
    secret = os.urandom(curve.key_size_bytes)
    keystream = _derive_keystream(curve, secret, len(data))
    ciphertext = bytes(d ^ k for d, k in zip(data, keystream))
    return secret + ciphertext


def decrypt(encrypted_data: bytes, curve_name: str) -> bytes:
    """Decifra i dati protetti con :func:`encrypt`."""

    curve = _get_curve(curve_name)
    if len(encrypted_data) < curve.key_size_bytes:
        raise ValueError("Dati cifrati troppo corti per la curva scelta")

    secret = encrypted_data[: curve.key_size_bytes]
    ciphertext = encrypted_data[curve.key_size_bytes :]
    keystream = _derive_keystream(curve, secret, len(ciphertext))
    return bytes(c ^ k for c, k in zip(ciphertext, keystream))


def authenticate(data: bytes, curve_name: str) -> bytes:
    """Autentica i dati usando una curva di Koblitz simulata.

    Genera un segreto effimero della dimensione della curva e calcola un tag
    HMAC-SHA256. Il segreto viene prefissato al payload per consentire la
    verifica.
    """

    curve = _get_curve(curve_name)
    secret = os.urandom(curve.key_size_bytes)
    auth_key = _derive_keystream(curve, secret, hashes.SHA256().digest_size)
    hmac_ctx = hmac.HMAC(auth_key, hashes.SHA256())
    hmac_ctx.update(data)
    tag = hmac_ctx.finalize()
    return secret + data + tag


def verify(authenticated_data: bytes, curve_name: str) -> bytes:
    """Verifica l'autenticazione creata da :func:`authenticate`."""

    curve = _get_curve(curve_name)
    tag_len = hashes.SHA256().digest_size
    if len(authenticated_data) < curve.key_size_bytes + tag_len:
        raise ValueError("Dati autenticati troppo corti per la curva scelta")

    secret = authenticated_data[: curve.key_size_bytes]
    tag = authenticated_data[-tag_len:]
    data = authenticated_data[curve.key_size_bytes:-tag_len]
    auth_key = _derive_keystream(curve, secret, tag_len)
    hmac_ctx = hmac.HMAC(auth_key, hashes.SHA256())
    hmac_ctx.update(data)
    try:
        hmac_ctx.verify(tag)
    except InvalidSignature as exc:
        raise ValueError("Autenticazione Koblitz non valida") from exc
    return data
