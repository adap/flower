# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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


import os
from concurrent.futures import ThreadPoolExecutor

from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Util.Padding import pad, unpad


def create_shares(secret: bytes, threshold: int, num: int) -> list[bytes]:
    """Return a list of shares (bytes).

    Shares are created from the provided secret using Shamir's secret sharing.
    """
    # Shamir's secret sharing requires the secret to be a multiple of 16 bytes
    # (AES block size). Pad the secret to the next multiple of 16 bytes.
    secret_padded = pad(secret, 16)
    chunks = [secret_padded[i : i + 16] for i in range(0, len(secret_padded), 16)]

    # The share list should contain shares of the secret, and each share consists of:
    # <4 bytes of index><share of chunk1><share of chunk2>...<share of chunkN>
    share_list: list[bytearray] = [bytearray() for _ in range(num)]

    # Create shares for each chunk in parallel
    max_workers = min(len(chunks), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_shares in executor.map(
            lambda chunk: _shamir_split(threshold, num, chunk), chunks
        ):
            for idx, share in chunk_shares:
                # Initialize the share with the index if it is empty
                if not share_list[idx - 1]:
                    share_list[idx - 1] += idx.to_bytes(4, "little", signed=False)

                # Append the share to the bytes
                share_list[idx - 1] += share

    return [bytes(share) for share in share_list]


def _shamir_split(threshold: int, num: int, chunk: bytes) -> list[tuple[int, bytes]]:
    """Create shares for a chunk using Shamir's secret sharing.

    Each share is a tuple (index, share_bytes), where share_bytes is 16 bytes long.
    """
    return Shamir.split(threshold, num, chunk, ssss=False)


def combine_shares(share_list: list[bytes]) -> bytes:
    """Reconstruct the secret from a list of shares."""
    # Compute the number of chunks
    # Each share contains 4 bytes of index and 16 bytes of share for each chunk
    chunk_num = (len(share_list[0]) - 4) >> 4

    secret_padded = bytearray(0)
    chunk_shares_list: list[list[tuple[int, bytes]]] = [[] for _ in range(chunk_num)]

    # Split shares into chunks
    for share in share_list:
        # The first 4 bytes are the index
        index = int.from_bytes(share[:4], "little", signed=False)
        for i in range(chunk_num):
            start = (i << 4) + 4
            chunk_shares_list[i].append((index, share[start : start + 16]))

    # Combine shares for each chunk in parallel
    max_workers = min(chunk_num, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk in executor.map(_shamir_combine, chunk_shares_list):
            secret_padded += chunk

    try:
        secret = unpad(bytes(secret_padded), 16)
    except ValueError:
        # If unpadding fails, it means the shares are not valid
        raise ValueError("Failed to combine shares") from None
    return secret


def _shamir_combine(shares: list[tuple[int, bytes]]) -> bytes:
    """Reconstruct a chunk from shares using Shamir's secret sharing."""
    return Shamir.combine(shares, ssss=False)
