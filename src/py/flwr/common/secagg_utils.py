from typing import List, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
import base64
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.SecretSharing import Shamir
from concurrent.futures import ThreadPoolExecutor
import os


def generate_key_pairs():
    sk = ec.generate_private_key(ec.SECP384R1())
    pk = sk.public_key()
    return sk, pk


def private_key_to_bytes(sk: ec.EllipticCurvePrivateKey) -> bytes:
    return sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption,
    )


def bytes_to_private_key(b: bytes) -> ec.EllipticCurvePrivateKey:
    return serialization.load_pem_private_key(data=b, password=None)


def public_key_to_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
    return pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def bytes_to_public_key(b: bytes) -> ec.EllipticCurvePublicKey:
    return serialization.load_pem_public_key(data=b)


def generate_shared_key(
    sk: ec.EllipticCurvePrivateKey, pk: ec.EllipticCurvePublicKey
) -> bytes:
    # Generate a 32 byte urlsafe(for fernet) shared key from own private key and another public key
    sharedk = sk.exchange(ec.ECDH(), pk)
    derivedk = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
    ).derive(sharedk)
    return base64.urlsafe_b64encode(derivedk)


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    # key must be url safe
    f = Fernet(key)
    return f.encrypt(plaintext)


def decrypt(key: bytes, token: bytes):
    # key must be url safe
    f = Fernet(key)
    return f.decrypt(token)


def create_shares(
    secret: bytes, threshold: int, num: int
) -> List[List[Tuple[int, bytes]]]:
    # return list of list for each user. Each sublist contains a share for a 16 byte chunk of the secret.
    # The int part of the tuple represents the index of the share, not the index of the chunk it is representing.
    secret_padded = pad(secret, 16)
    secret_padded_chunk = [
        (threshold, num, secret_padded[i : i + 16])
        for i in range(0, len(secret_padded), 16)
    ]
    share_list = []
    for i in range(num):
        share_list.append([])

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk_shares in executor.map(
            lambda arg: shamir_split(*arg), secret_padded_chunk
        ):
            for idx, share in chunk_shares:
                # idx start with 1
                share_list[idx - 1].append((idx, share))
    return share_list


def shamir_split(threshold: int, num: int, chunk: bytes):
    return Shamir.split(threshold, num, chunk)


def combine_shares(shares: List[List[Tuple[int, bytes]]]):
    chunk_num = len(shares[0])
    secret_padded = bytearray(0)
    chunk_shares_list = []
    for i in range(chunk_num):
        chunk_shares = []
        for j in range(len(shares)):
            chunk_shares.append(shares[j][i])
        chunk_shares_list.append(chunk_shares)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in executor.map(shamir_combine, chunk_shares_list):
            secret_padded += chunk

    secret = unpad(secret_padded, 16)
    return bytes(secret)


def shamir_combine(shares: List[Tuple[int, bytes]]):
    return Shamir.combine(shares)


def random_bytes(num: int = 32):
    return os.random(num)
