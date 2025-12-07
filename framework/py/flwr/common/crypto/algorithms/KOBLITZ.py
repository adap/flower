"""Implementazioni semplificate di curve di Koblitz per la demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from cryptography.hazmat.primitives import hashes
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
    "KOBLITZ_112": KoblitzCurve("KOBLITZ_112", 112),
    "KOBLITZ_256": KoblitzCurve("KOBLITZ_256", 256),
    "KOBLITZ_512": KoblitzCurve("KOBLITZ_512", 512),
}


def _get_curve(curve_name: str) -> KoblitzCurve:
    try:
        return SUPPORTED_CURVES[curve_name]
    except KeyError as exc:  # pragma: no cover - safety net
        raise ValueError(f"Curva Koblitz non supportata: {curve_name}") from exc


def _derive_keystream(curve: KoblitzCurve, secret: bytes, length: int) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=curve.name.encode(),
    )
    return hkdf.derive(secret)


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
