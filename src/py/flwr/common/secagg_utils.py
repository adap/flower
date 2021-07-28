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
import random
import pickle
import numpy as np

from numpy.core.fromnumeric import clip

from flwr.common.typing import Weights

# Key Generation


def generate_key_pairs():
    sk = ec.generate_private_key(ec.SECP384R1())
    pk = sk.public_key()
    return sk, pk


def private_key_to_bytes(sk: ec.EllipticCurvePrivateKey) -> bytes:
    return sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
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

# Authenticated Encryption


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    # key must be url safe
    f = Fernet(key)
    return f.encrypt(plaintext)


def decrypt(key: bytes, token: bytes):
    # key must be url safe
    f = Fernet(key)
    return f.decrypt(token)

# Shamir's Secret Sharing Scheme


def create_shares(
    secret: bytes, threshold: int, num: int
) -> List[bytes]:
    # return list of list for each user. Each sublist contains a share for a 16 byte chunk of the secret.
    # The int part of the tuple represents the index of the share, not the index of the chunk it is representing.
    secret_padded = pad(secret, 16)
    secret_padded_chunk = [
        (threshold, num, secret_padded[i: i + 16])
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

    for idx, shares in enumerate(share_list):
        share_list[idx] = pickle.dumps(shares)

    return share_list


def shamir_split(threshold: int, num: int, chunk: bytes):
    return Shamir.split(threshold, num, chunk)


def combine_shares(share_list: List[bytes]):
    for idx, share in enumerate(share_list):
        share_list[idx] = pickle.loads(share)

    print(share_list)
    chunk_num = len(share_list[0])
    secret_padded = bytearray(0)
    chunk_shares_list = []
    for i in range(chunk_num):
        chunk_shares = []
        for j in range(len(share_list)):
            chunk_shares.append(share_list[j][i])
        chunk_shares_list.append(chunk_shares)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in executor.map(shamir_combine, chunk_shares_list):
            secret_padded += chunk

    secret = unpad(secret_padded, 16)
    return bytes(secret)


def shamir_combine(shares: List[Tuple[int, bytes]]):
    return Shamir.combine(shares)


# Random Bytes Generator
def rand_bytes(num: int = 32) -> bytes:
    return os.urandom(num)

# Pseudo Bytes Generator


def pseudo_rand_gen(seed: bytes, num_range: int, l: int):
    random.seed(seed)
    return [random.randrange(0, num_range) for i in range(l)]


# String Concatenation
def share_keys_plaintext_concat(source: int, destination: int, b_share: bytes, sk_share: bytes):
    concat = b'||'
    return concat.join([str(source).encode(), str(destination).encode(), b_share, sk_share])


def share_keys_plaintext_separate(plaintext: bytes):
    plaintext_list = plaintext.split(b"||")
    return (
        plaintext_list[0].decode("utf-8", "strict"),
        plaintext_list[1].decode("utf-8", "strict"),
        plaintext_list[2],
        plaintext_list[3],
    )

# Weight Quantization


def quantize(weight: Weights, clipping_range: float, target_range: int) -> Weights:
    quantized_list = []
    f = np.vectorize(lambda x:  min(target_range-1, (sorted((-clipping_range, x, clipping_range))
                                                     [1]+clipping_range)*target_range/(2*clipping_range)))
    for arr in weight:
        quantized_list.append(f(arr).astype(int))
    return quantized_list


def reverse_quantize(weight: Weights, clipping_range: float, target_range: int) -> Weights:
    reverse_quantized_list = []
    f = np.vectorize(lambda x:  (x)/target_range*(2*clipping_range)-clipping_range)
    for arr in weight:
        reverse_quantized_list.append(f(arr.astype(float)))
    return reverse_quantized_list
