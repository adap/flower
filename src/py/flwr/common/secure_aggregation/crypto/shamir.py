# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Shamir's secret sharing."""


import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, cast

from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Util.Padding import pad, unpad


def create_shares(secret: bytes, threshold: int, num: int) -> List[bytes]:
    """Return list of shares (bytes)."""
    secret_padded = pad(secret, 16)
    secret_padded_chunk = [
        (threshold, num, secret_padded[i : i + 16])
        for i in range(0, len(secret_padded), 16)
    ]
    share_list: List[List[Tuple[int, bytes]]] = [[] for _ in range(num)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk_shares in executor.map(
            lambda arg: _shamir_split(*arg), secret_padded_chunk
        ):
            for idx, share in chunk_shares:
                # Index in `chunk_shares` starts from 1
                share_list[idx - 1].append((idx, share))

    return [pickle.dumps(shares) for shares in share_list]


def _shamir_split(threshold: int, num: int, chunk: bytes) -> List[Tuple[int, bytes]]:
    return Shamir.split(threshold, num, chunk, ssss=False)


# Reconstructing secret with PyCryptodome
def combine_shares(share_list: List[bytes]) -> bytes:
    """Reconstruct secret from shares."""
    unpickled_share_list: List[List[Tuple[int, bytes]]] = [
        cast(List[Tuple[int, bytes]], pickle.loads(share)) for share in share_list
    ]

    chunk_num = len(unpickled_share_list[0])
    secret_padded = bytearray(0)
    chunk_shares_list: List[List[Tuple[int, bytes]]] = []
    for i in range(chunk_num):
        chunk_shares: List[Tuple[int, bytes]] = []
        for share in unpickled_share_list:
            chunk_shares.append(share[i])
        chunk_shares_list.append(chunk_shares)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in executor.map(_shamir_combine, chunk_shares_list):
            secret_padded += chunk

    secret = unpad(secret_padded, 16)
    return bytes(secret)


def _shamir_combine(shares: List[Tuple[int, bytes]]) -> bytes:
    return Shamir.combine(shares, ssss=False)
