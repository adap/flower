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


from collections.abc import MutableMapping
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    TypeVar,
    Union,
    ValuesView,
    cast,
    overload,
)

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type
_T = TypeVar("_T")


class TypedDict(MutableMapping[K, V], Generic[K, V]):
    """Typed dictionary."""

    def __init__(
        self, check_key_fn: Callable[[K], None], check_value_fn: Callable[[V], None]
    ):
        self.__dict__["_check_key_fn"] = check_key_fn
        self.__dict__["_check_value_fn"] = check_value_fn
        self.__dict__["_data"] = {}

    def __setitem__(self, key: K, value: V) -> None:
        """Set the given key to the given value after type checking."""
        # Check the types of key and value
        cast(Callable[[K], None], self.__dict__["_check_key_fn"])(key)
        cast(Callable[[V], None], self.__dict__["_check_value_fn"])(value)

        # Set key-value pair
        cast(Dict[K, V], self.__dict__["_data"])[key] = value

    def __delitem__(self, key: K) -> None:
        """Remove the item with the specified key."""
        del cast(Dict[K, V], self.__dict__["_data"])[key]

    def __getitem__(self, item: K) -> V:
        """Return the value for the specified key."""
        return cast(Dict[K, V], self.__dict__["_data"])[item]

    def __iter__(self) -> Iterator[K]:
        """Yield an iterator over the keys of the dictionary."""
        return iter(cast(Dict[K, V], self.__dict__["_data"]))

    def __repr__(self) -> str:
        """Return a string representation of the dictionary."""
        return cast(Dict[K, V], self.__dict__["_data"]).__repr__()

    def __len__(self) -> int:
        """Return the number of items in the dictionary."""
        return len(cast(Dict[K, V], self.__dict__["_data"]))

    def __contains__(self, key: object) -> bool:
        """Check if the dictionary contains the specified key."""
        return key in cast(Dict[K, V], self.__dict__["_data"])

    def __eq__(self, other: object) -> bool:
        """Compare this instance to another dictionary or TypedDict."""
        data = cast(Dict[K, V], self.__dict__["_data"])
        if isinstance(other, TypedDict):
            other_data = cast(Dict[K, V], other.__dict__["_data"])
            return data == other_data
        if isinstance(other, dict):
            return data == other
        return NotImplemented

    def items(self) -> ItemsView[K, V]:
        """R.items() -> a set-like object providing a view on R's items."""
        return cast(Dict[K, V], self.__dict__["_data"]).items()

    def keys(self) -> KeysView[K]:
        """R.keys() -> a set-like object providing a view on R's keys."""
        return cast(Dict[K, V], self.__dict__["_data"]).keys()

    def values(self) -> ValuesView[V]:
        """R.values() -> an object providing a view on R's values."""
        return cast(Dict[K, V], self.__dict__["_data"]).values()
    
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /, **kwargs: V) -> None: ...
    

    def update(self, m: Any = None, **kwargs: Any) -> None:
        """R.update([E, ]**F) -> None.

        Update R from dict/iterable E and F.
        """
        for key, value in dict(m, **kwargs).items():
            self[key] = value

    @overload
    def pop(self, key: K, /) -> V: ...

    @overload
    def pop(self, key: K, /, default: V) -> V: ...

    @overload
    def pop(self, key: K, /, default: _T) -> Union[V, _T]: ...

    def pop(self, key: K, /, default: Any = None) -> Any:
        """R.pop(k[,d]) -> v, remove specified key and return the corresponding value.

        If key is not found, d is returned if given, otherwise KeyError is raised.
        """
        if default is None:
            return cast(Dict[K, V], self.__dict__["_data"]).pop(key)
        else:
            return cast(Dict[K, V], self.__dict__["_data"]).pop(key, default)

    @overload
    def get(self, key: K, /) -> Optional[V]: ...

    @overload
    def get(self, key: K, /, default: Union[V, _T]) -> Union[V, _T]: ...

    def get(self, key: K, /, default: Any = None) -> Any:
        """R.get(k[,d]) -> R[k] if k in R, else d.

        d defaults to None.
        """
        if default is None:
            return cast(Dict[K, V], self.__dict__["_data"]).get(key)
        else:
            return cast(Dict[K, V], self.__dict__["_data"]).get(key, default)

    def clear(self) -> None:
        """R.clear() -> None.

        Remove all items from R.
        """
        cast(Dict[K, V], self.__dict__["_data"]).clear()
