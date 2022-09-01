# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import division
from __future__ import print_function
import math
from logging import WARNING, log
import functools
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
import json

from flwr.common.typing import Weights

# String Concatenation ===================================================================


# Unambiguous string concatenation of source, destination, and two secret shares.
# We assume they do not contain the 'abcdef' string
def share_keys_plaintext_concat(source: int, destination: int, b_share: bytes, sk_share: bytes) -> bytes:
    # return pickle.dumps([source, destination, b_share, sk_share])
    # concat = b'abcdefg'
    # return b'abcdefg'.join([str(source).encode(), str(destination).encode(), b_share, sk_share])
    source, destination = int.to_bytes(source, 4, 'little'), int.to_bytes(destination, 4, 'little')
    return b''.join([source, destination, int.to_bytes(len(b_share), 4, 'little'), b_share, sk_share])

# Unambiguous string splitting to obtain source, destination and two secret shares.


def share_keys_plaintext_separate(plaintext: bytes) -> Tuple[int, int, bytes, bytes]:
    # l = pickle.loads(plaintext)
    # return tuple(l)
    # plaintext_list = plaintext.split(b"abcdefg")
    # return (
    #     int(plaintext_list[0].decode("utf-8", "strict")),
    #     int(plaintext_list[1].decode("utf-8", "strict")),
    #     plaintext_list[2],
    #     plaintext_list[3],
    # )
    src, dst, mark = int.from_bytes(plaintext[:4], 'little'), int.from_bytes(plaintext[4:8], 'little'), \
        int.from_bytes(plaintext[8:12], 'little')
    ret = [src, dst, plaintext[12:12 + mark], plaintext[12 + mark:]]
    return ret


# # Key Generation  ====================================================================
#
# # Generate private and public key pairs with Cryptography
#
#
# def generate_key_pairs() -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
#     sk = ec.generate_private_key(ec.SECP384R1())
#     pk = sk.public_key()
#     return sk, pk
#
# # Serialize private key
#
#
# def private_key_to_bytes(sk: ec.EllipticCurvePrivateKey) -> bytes:
#     return sk.private_bytes(
#         encoding=serialization.Encoding.PEM,
#         format=serialization.PrivateFormat.PKCS8,
#         encryption_algorithm=serialization.NoEncryption(),
#     )
#
# # Deserialize private key
#
#
# def bytes_to_private_key(b: bytes) -> ec.EllipticCurvePrivateKey:
#     return serialization.load_pem_private_key(data=b, password=None)
#
# # Serialize public key
#
#
# def public_key_to_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
#     return pk.public_bytes(
#         encoding=serialization.Encoding.PEM,
#         format=serialization.PublicFormat.SubjectPublicKeyInfo,
#     )
#
# # Deserialize public key
#
#
# def bytes_to_public_key(b: bytes) -> ec.EllipticCurvePublicKey:
#     return serialization.load_pem_public_key(data=b)
#
# # Generate shared key by exchange function and key derivation function
# # Key derivation function is needed to obtain final shared key of exactly 32 bytes
#
#
# def generate_shared_key(
#     sk: ec.EllipticCurvePrivateKey, pk: ec.EllipticCurvePublicKey
# ) -> bytes:
#     # Generate a 32 byte urlsafe(for fernet) shared key from own private key and another public key
#     sharedk = sk.exchange(ec.ECDH(), pk)
#     derivedk = HKDF(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=None,
#         info=None,
#     ).derive(sharedk)
#     return base64.urlsafe_b64encode(derivedk)

# # Authenticated Encryption ================================================================
#
# # Encrypt plaintext with Fernet. Key must be 32 bytes.
#
#
# def encrypt(key: bytes, plaintext: bytes) -> bytes:
#     # key must be url safe
#     f = Fernet(key)
#     return f.encrypt(plaintext)
#
# # Decrypt ciphertext with Fernet. Key must be 32 bytes.
#
#
# def decrypt(key: bytes, token: bytes):
#     # key must be url safe
#     f = Fernet(key)
#     return f.decrypt(token)

# Shamir's Secret Sharing Scheme ============================================================


# Create shares with PyCryptodome. Each share must be processed to be a byte string with pickle for RPC


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
    #print("send", [len(i) for i in share_list])

    return share_list


def shamir_split(threshold: int, num: int, chunk: bytes) -> List[Tuple[int, bytes]]:
    return Shamir.split(threshold, num, chunk)

# Reconstructing secret with PyCryptodome


def combine_shares(share_list: List[bytes]) -> bytes:
    #print("receive", [len(i) for i in share_list])
    for idx, share in enumerate(share_list):
        share_list[idx] = pickle.loads(share)

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


def shamir_combine(shares: List[Tuple[int, bytes]]) -> bytes:
    return Shamir.combine(shares)


# Random Bytes Generator =============================================================

# Generate random bytes with os. Usually 32 bytes for Fernet
def rand_bytes(num: int = 32) -> bytes:
    return os.urandom(num)

# Pseudo Bytes Generator ==============================================================

# Pseudo random generator for creating masks.

# the one use numpy PRG
def pseudo_rand_gen(seed: bytes, num_range: int, dimensions_list: List[Tuple]) -> Weights:
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i:i+4], 'little')
    np.random.seed(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(np.random.randint(0, num_range - 1), dtype=int)
        else:
            arr = np.random.randint(0, num_range - 1, dimension)
        output.append(arr)
    return output


# # Weight Manipulation =============================================================
#
# # Combine factor with weights
#
#
# def factor_weights_combine(weights_factor: int, weights: Weights) -> Weights:
#     return [np.array([weights_factor])]+weights
#
# # Extract factor from weights
#
#
# def factor_weights_extract(weights: Weights) -> Tuple[int, Weights]:
#     return weights[0][0], weights[1:]
#
# # Create dimensions list of each element in weights
#
#
# def weights_shape(weights: Weights) -> List[Tuple]:
#     return [arr.shape for arr in weights]
#
# # Generate zero weights based on dimensions list
#
#
# def weights_zero_generate(dimensions_list: List[Tuple], dtype=np.int64) -> Weights:
#     return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]
#
# # Add two weights together
#
#
# def weights_addition(a: Weights, b: Weights) -> Weights:
#     return [a[idx]+b[idx] for idx in range(len(a))]
#
# # Subtract one weight from the other
#
#
# def weights_subtraction(a: Weights, b: Weights) -> Weights:
#     return [a[idx]-b[idx] for idx in range(len(a))]
#
# # Take mod of a weights with an integer
#
#
# def weights_mod(a: Weights, b: int) -> Weights:
#     if bin(b).count("1") == 1:
#         msk = b - 1
#         return [a[idx] & msk for idx in range(len(a))]
#     return [a[idx] % b for idx in range(len(a))]
#
#
# # Multiply weight by an integer
#
#
# def weights_multiply(a: Weights, b: int) -> Weights:
#     return [a[idx] * b for idx in range(len(a))]
#
# # Divide weight by an integer
#
#
# def weights_divide(a: Weights, b: int) -> Weights:
#     return [a[idx] / b for idx in range(len(a))]
