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


class NoObjectInStoreError(Exception):
    """Error when trying to access an element in the ObjectStore that does not exist."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Return formatted exception message string."""
        return f"NoObjectInStoreError: {self.message}"


class ObjectStore(abc.ABC):
    """Abstract base class for `ObjectStore` implementations.

    This class defines the interface for an object store that can store, retrieve, and
    delete objects identified by object IDs.
    """

    @abc.abstractmethod
    def preregister(self, object_ids: list[str]) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`.

        Parameters
        ----------
        object_ids : list[str]
            A list of object IDs to check against the store. Any object ID not already
            present will be preregistered.

        Returns
        -------
        list[str]
            A list of object IDs that were not present in the `ObjectStore` and have now
            been preregistered.
        """

    @abc.abstractmethod
    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store.

        Parameters
        ----------
        object_id : str
            The object_id under which to store the object. Must be preregistered.
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
    def set_message_descendant_ids(
        self, msg_object_id: str, descendant_ids: list[str]
    ) -> None:
        """Store the mapping from a ``Message`` object ID to the object IDs of its
        descendants.

        Parameters
        ----------
        msg_object_id : str
            The object ID of the ``Message``.
        descendant_ids : list[str]
            A list of object IDs representing all descendant objects of the ``Message``.
        """

    @abc.abstractmethod
    def get_message_descendant_ids(self, msg_object_id: str) -> list[str]:
        """Retrieve the object IDs of all descendants of a given ``Message``.

        Parameters
        ----------
        msg_object_id : str
            The object ID of the ``Message``.

        Returns
        -------
        list[str]
            A list of object IDs of all descendant objects of the ``Message``.
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
