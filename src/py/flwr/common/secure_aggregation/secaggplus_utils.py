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
"""Utility functions for the SecAgg/SecAgg+ protocol."""


from typing import List, Tuple

import numpy as np

from flwr.common.typing import NDArrayInt


def share_keys_plaintext_concat(
    source: int, destination: int, b_share: bytes, sk_share: bytes
) -> bytes:
    """Combine arguments to bytes.

    Parameters
    ----------
    source : int
        the secure ID of the source.
    destination : int
        the secure ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.

    Returns
    -------
    bytes
        The combined bytes of all the arguments.
    """
    return b"".join(
        [
            int.to_bytes(source, 4, "little"),
            int.to_bytes(destination, 4, "little"),
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


def share_keys_plaintext_separate(plaintext: bytes) -> Tuple[int, int, bytes, bytes]:
    """Retrieve arguments from bytes.

    Parameters
    ----------
    plaintext : bytes
        the bytes containing 4 arguments.

    Returns
    -------
    source : int
        the secure ID of the source.
    destination : int
        the secure ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.
    """
    src, dst, mark = (
        int.from_bytes(plaintext[:4], "little"),
        int.from_bytes(plaintext[4:8], "little"),
        int.from_bytes(plaintext[8:12], "little"),
    )
    ret = (src, dst, plaintext[12 : 12 + mark], plaintext[12 + mark :])
    return ret


def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: List[Tuple[int, ...]]
) -> List[NDArrayInt]:
    """Seeded pseudo-random number generator for noise generation with Numpy."""
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i : i + 4], "little")
    # pylint: disable-next=no-member
    gen = np.random.RandomState(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(gen.randint(0, num_range - 1), dtype=int)
        else:
            arr = gen.randint(0, num_range - 1, dimension)
        output.append(arr)
    return output
