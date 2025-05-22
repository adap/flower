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


import abc
from typing import Optional


class ObjectStore(abc.ABC):
    """Abstract base class for `ObjectStore` implementations.

    This class defines the interface for an object store that can store, retrieve, and
    delete objects identified by object IDs.
    """

    @abc.abstractmethod
    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store.

        Parameters
        ----------
        object_id : str
            The object_id under which to store the object.
        object_content : bytes
            The deflated object to store.
        """

    @abc.abstractmethod
    def get(self, object_id: str) -> Optional[bytes]:
        """Get an object from the store.

        Parameters
        ----------
        object_id : str
            The object_id under which the object is stored.

        Returns
        -------
        bytes
            The object stored under the given object_id.
        """

    @abc.abstractmethod
    def delete(self, object_id: str) -> None:
        """Delete an object from the store.

        Parameters
        ----------
        object_id : str
            The object_id under which the object is stored.
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear the store.

        This method should remove all objects from the store.
        """

    @abc.abstractmethod
    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store.

        Parameters
        ----------
        object_id : str
            The object_id to check.

        Returns
        -------
        bool
            True if the object_id is in the store, False otherwise.
        """
