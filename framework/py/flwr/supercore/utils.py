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
"""Utility functions for the infrastructure."""


def mask_string(value: str, head: int = 4, tail: int = 4) -> str:
    """Mask a string by preserving only the head and tail characters.

    Mask a string for safe display by preserving the head and tail characters,
    and replacing the middle with '...'. Useful for logging tokens, secrets,
    or IDs without exposing sensitive data.

    Notes
    -----
    If the string is shorter than the combined length of `head` and `tail`,
    the original string is returned unchanged.
    """
    if len(value) <= head + tail:
        return value
    return f"{value[:head]}...{value[-tail:]}"


def uint64_to_int64(unsigned: int) -> int:
    """Convert a uint64 integer to a sint64 with the same bit pattern.

    For values >= 2^63, wraps around by subtracting 2^64.
    """
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


def int64_to_uint64(signed: int) -> int:
    """Convert a sint64 integer to a uint64 with the same bit pattern.

    For negative values, wraps around by adding 2^64.
    """
    if signed < 0:
        return signed + (1 << 64)
    return signed
