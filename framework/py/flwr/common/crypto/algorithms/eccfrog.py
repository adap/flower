"""Implementazione della curva ECCFROG522PP e delle firme ECDSA."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


P = int(
    "686479766013060971498190079908139321726943530014330540939446345918554318"
    "339765605212255964066145455497729631139148085803712198799971664381257402"
    "8291115058039"
)
A = -9 % P
B = int(
    "661139136184195850860452469937744791138999490012975421307768311225096419"
    "509388251093415492337101182055425457255989613682399356563300695566619742"
    "8760619911"
)
N = int(
    "686479766013060971498190079908139321726943530014330540939446345918554318"
    "339765470783993099806907243717889863432321841973824511791072608043490749"
    "5541251156283"
)
GX = int(
    "114836598700559139646235363713136312609767670986199491984058026550790121"
    "317888159000151000981405923011587990724012666535482931446873066751491073"
    "89798128134"
)
GY = int(
    "303869445742844202438813211737067794312734393851211346303431863870960045"
    "113632574702513861080239149191409127648110569935391920249490281068659303"
    "0172286395020"
)

KEY_SIZE_BITS = 521
KEY_SIZE_BYTES = (KEY_SIZE_BITS + 7) // 8


@dataclass(frozen=True)
class ECPoint:
    """Rappresentazione di un punto affine sulla curva."""

    x: int
    y: int

    def to_bytes(self) -> bytes:
        return b"\x04" + _int_to_bytes(self.x) + _int_to_bytes(self.y)


@dataclass(frozen=True)
class ECCFROGPublicKey:
    point: ECPoint

    def to_bytes(self) -> bytes:
        return self.point.to_bytes()


@dataclass(frozen=True)
class ECCFROGPrivateKey:
    secret: int
    public_key: ECCFROGPublicKey

    def to_bytes(self) -> bytes:
        return _int_to_bytes(self.secret)


def generate_private_key() -> ECCFROGPrivateKey:
    secret = secrets.randbelow(N - 1) + 1
    public_point = scalar_mult(secret, ECPoint(GX, GY))
    if public_point is None:  # pragma: no cover - sicurezza
        raise ValueError("Punto pubblico non valido")
    return ECCFROGPrivateKey(secret, ECCFROGPublicKey(public_point))


def load_private_key(key: object) -> ECCFROGPrivateKey:
    if isinstance(key, ECCFROGPrivateKey):
        return key
    if isinstance(key, int):
        secret = key
    elif isinstance(key, (bytes, bytearray)):
        secret = int.from_bytes(bytes(key), "big")
    elif isinstance(key, str):
        key_str = key.strip().lower()
        if key_str.startswith("0x"):
            key_str = key_str[2:]
        secret = int.from_bytes(bytes.fromhex(key_str), "big")
    else:
        raise ValueError("Formato chiave privata ECCFROG non supportato")
    if not (1 <= secret < N):
        raise ValueError("Chiave privata ECCFROG fuori range")
    public_point = scalar_mult(secret, ECPoint(GX, GY))
    if public_point is None:  # pragma: no cover - sicurezza
        raise ValueError("Punto pubblico non valido")
    return ECCFROGPrivateKey(secret, ECCFROGPublicKey(public_point))


def load_public_key(key: object) -> ECCFROGPublicKey:
    if isinstance(key, ECCFROGPublicKey):
        return key
    if isinstance(key, ECPoint):
        return ECCFROGPublicKey(key)
    if isinstance(key, (bytes, bytearray)):
        key_bytes = bytes(key)
    elif isinstance(key, str):
        key_str = key.strip().lower()
        if key_str.startswith("0x"):
            key_str = key_str[2:]
        key_bytes = bytes.fromhex(key_str)
    else:
        raise ValueError("Formato chiave pubblica ECCFROG non supportato")
    if len(key_bytes) != 1 + 2 * KEY_SIZE_BYTES or key_bytes[0] != 0x04:
        raise ValueError("Chiave pubblica ECCFROG non valida")
    x = int.from_bytes(key_bytes[1 : 1 + KEY_SIZE_BYTES], "big")
    y = int.from_bytes(key_bytes[1 + KEY_SIZE_BYTES :], "big")
    point = ECPoint(x, y)
    if not is_on_curve(point):
        raise ValueError("Punto pubblico ECCFROG non sulla curva")
    return ECCFROGPublicKey(point)


def sign(message: bytes, private_key: ECCFROGPrivateKey) -> bytes:
    z = _hash_to_int(message)
    while True:
        k = secrets.randbelow(N - 1) + 1
        r_point = scalar_mult(k, ECPoint(GX, GY))
        if r_point is None:
            continue
        r = r_point.x % N
        if r == 0:
            continue
        k_inv = _modinv(k, N)
        s = (k_inv * (z + r * private_key.secret)) % N
        if s == 0:
            continue
        return _int_to_bytes(r, size=KEY_SIZE_BYTES) + _int_to_bytes(
            s, size=KEY_SIZE_BYTES
        )


def verify(message: bytes, signature: bytes, public_key: ECCFROGPublicKey) -> bool:
    if len(signature) != 2 * KEY_SIZE_BYTES:
        return False
    r = int.from_bytes(signature[:KEY_SIZE_BYTES], "big")
    s = int.from_bytes(signature[KEY_SIZE_BYTES:], "big")
    if not (1 <= r < N and 1 <= s < N):
        return False
    z = _hash_to_int(message)
    w = _modinv(s, N)
    u1 = (z * w) % N
    u2 = (r * w) % N
    point = point_add(
        scalar_mult(u1, ECPoint(GX, GY)),
        scalar_mult(u2, public_key.point),
    )
    if point is None:
        return False
    return (point.x % N) == r


def is_on_curve(point: ECPoint) -> bool:
    return (point.y * point.y - (point.x * point.x * point.x + A * point.x + B)) % P == 0


def scalar_mult(k: int, point: ECPoint | None) -> ECPoint | None:
    if point is None or k % N == 0:
        return None
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1
    return result


def point_add(p1: ECPoint | None, p2: ECPoint | None) -> ECPoint | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    if p1.x == p2.x and (p1.y != p2.y or p1.y == 0):
        return None
    if p1.x == p2.x:
        return point_double(p1)
    slope = ((p2.y - p1.y) * _modinv(p2.x - p1.x, P)) % P
    x3 = (slope * slope - p1.x - p2.x) % P
    y3 = (slope * (p1.x - x3) - p1.y) % P
    return ECPoint(x3, y3)


def point_double(point: ECPoint | None) -> ECPoint | None:
    if point is None:
        return None
    if point.y == 0:
        return None
    slope = ((3 * point.x * point.x + A) * _modinv(2 * point.y, P)) % P
    x3 = (slope * slope - 2 * point.x) % P
    y3 = (slope * (point.x - x3) - point.y) % P
    return ECPoint(x3, y3)


def _hash_to_int(message: bytes) -> int:
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest, "big") % N


def _int_to_bytes(value: int, size: int = KEY_SIZE_BYTES) -> bytes:
    return value.to_bytes(size, "big")


def _modinv(value: int, modulus: int) -> int:
    return pow(value, -1, modulus)
