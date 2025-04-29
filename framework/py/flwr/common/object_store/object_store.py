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
"""Flower abstract ObjectStore definition."""

from abc import ABC
from typing import Optional


class ObjectStore(ABC):
    """Abstract base class for `ObjectStore` implementations.

    This class defines the interface for an object store that can store, retrieve, and
    delete objects identified by keys.
    """

    def put(self, key: str, value: bytes) -> None:
        """Put an object into the store.

        Parameters
        ----------
        key : str
            The key under which to store the object.
        value : bytes
            The serialized object to store.
        """
        raise NotImplementedError

    def get(self, key: str) -> Optional[bytes]:
        """Get an object from the store.

        Parameters
        ----------
        key : str
            The key under which the object is stored.

        Returns
        -------
        bytes
            The object stored under the given key.
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete an object from the store.

        Parameters
        ----------
        key : str
            The key under which the object is stored.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear the store.

        This method should remove all objects from the store.
        """
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the store.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
            True if the key is in the store, False otherwise.
        """
        raise NotImplementedError
