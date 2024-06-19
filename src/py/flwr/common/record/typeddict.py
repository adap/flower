# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Typed dict base class for *Records."""


from typing import Any, Callable, Dict, Generic, Iterator, Tuple, TypeVar, cast

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class TypedDict(Generic[K, V]):
    """Typed dictionary."""

    def __init__(
        self, check_key_fn: Callable[[K], None], check_value_fn: Callable[[V], None]
    ):
        self._data: Dict[K, V] = {}
        self._check_key_fn = check_key_fn
        self._check_value_fn = check_value_fn

    def __setitem__(self, key: K, value: V) -> None:
        """Set the given key to the given value after type checking."""
        # Check the types of key and value
        self._check_key_fn(key)
        self._check_value_fn(value)
        # Set key-value pair
        self._data[key] = value

    def __delitem__(self, key: K) -> None:
        """Remove the item with the specified key."""
        del self._data[key]

    def __getitem__(self, item: K) -> V:
        """Return the value for the specified key."""
        return self._data[item]

    def __iter__(self) -> Iterator[K]:
        """Yield an iterator over the keys of the dictionary."""
        return iter(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the dictionary."""
        return self._data.__repr__()

    def __len__(self) -> int:
        """Return the number of items in the dictionary."""
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        """Check if the dictionary contains the specified key."""
        return key in self._data

    def __eq__(self, other: object) -> bool:
        """Compare this instance to another dictionary or TypedDict."""
        if isinstance(other, TypedDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def items(self) -> Iterator[Tuple[K, V]]:
        """R.items() -> a set-like object providing a view on R's items."""
        return cast(Iterator[Tuple[K, V]], self._data.items())

    def keys(self) -> Iterator[K]:
        """R.keys() -> a set-like object providing a view on R's keys."""
        return cast(Iterator[K], self._data.keys())

    def values(self) -> Iterator[V]:
        """R.values() -> an object providing a view on R's values."""
        return cast(Iterator[V], self._data.values())

    def update(self, *args: Any, **kwargs: Any) -> None:
        """R.update([E, ]**F) -> None.

        Update R from dict/iterable E and F.
        """
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def pop(self, key: K) -> V:
        """R.pop(k[,d]) -> v, remove specified key and return the corresponding value.

        If key is not found, d is returned if given, otherwise KeyError is raised.
        """
        return self._data.pop(key)

    def get(self, key: K, default: V) -> V:
        """R.get(k[,d]) -> R[k] if k in R, else d.

        d defaults to None.
        """
        return self._data.get(key, default)

    def clear(self) -> None:
        """R.clear() -> None.

        Remove all items from R.
        """
        self._data.clear()