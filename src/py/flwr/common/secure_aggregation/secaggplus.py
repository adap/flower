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
"""Utility functions for the SecAgg/SecAgg+ protocol."""


from typing import List, Tuple

import numpy as np


# Unambiguous string concatenation of source, destination, and two secret shares.
# We assume they do not contain the 'abcdef' string
def share_keys_plaintext_concat(
    source: int, destination: int, b_share: bytes, sk_share: bytes
) -> bytes:
    source, destination = int.to_bytes(source, 4, "little"), int.to_bytes(
        destination, 4, "little"
    )
    return b"".join(
        [
            source,
            destination,
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


# Unambiguous string splitting to obtain source, destination and two secret shares.


def share_keys_plaintext_separate(plaintext: bytes) -> Tuple[int, int, bytes, bytes]:
    src, dst, mark = (
        int.from_bytes(plaintext[:4], "little"),
        int.from_bytes(plaintext[4:8], "little"),
        int.from_bytes(plaintext[8:12], "little"),
    )
    ret = (src, dst, plaintext[12 : 12 + mark], plaintext[12 + mark :])
    return ret


# Pseudo Bytes Generator ==============================================================


# Pseudo random generator for creating masks.
# the one use numpy PRG
def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: List[Tuple]
) -> List[np.ndarray]:
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i : i + 4], "little")
    gen = np.random.RandomState(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(gen.randint(0, num_range - 1), dtype=int)
        else:
            arr = gen.randint(0, num_range - 1, dimension)
        output.append(arr)
    return output
