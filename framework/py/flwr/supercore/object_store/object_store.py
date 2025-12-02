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

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611


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
    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`.

        Parameters
        ----------
        run_id : int
            The ID of the run for which to preregister objects.
        object_tree : ObjectTree
            The object tree containing the IDs of objects to preregister.
            This tree should contain all objects that are expected to be
            stored in the `ObjectStore`.

        Returns
        -------
        list[str]
            A list of object IDs that were either not previously preregistered
            in the `ObjectStore`, or were preregistered but are not yet available.
        """

    @abc.abstractmethod
    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID.

        Parameters
        ----------
        object_id : str
            The ID of the object for which to retrieve the object tree.

        Returns
        -------
        ObjectTree
            An ObjectTree representing the hierarchical structure of the object with
            the given ID and its descendants.
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
    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store.

        Parameters
        ----------
        object_id : str
            The object_id under which the object is stored.

        Returns
        -------
        Optional[bytes]
            The object stored under the given object_id if it exists, else None.
            The returned bytes will be b"" if the object is not yet available,
            but has been preregistered.
        """

    @abc.abstractmethod
    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store.

        This method attempts to recursively delete the specified object and its
        descendants, if they are not referenced by any other object.

        Parameters
        ----------
        object_id : str
            The object_id under which the object is stored.

        Notes
        -----
        The object of the given object_id will NOT be deleted if it is still referenced
        by any other object in the store.
        """

    @abc.abstractmethod
    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run.

        Parameters
        ----------
        run_id : int
            The ID of the run for which to delete objects.

        Notes
        -----
        Objects that are still registered in other runs will NOT be deleted.
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

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of objects in the store.

        Returns
        -------
        int
            The number of objects currently stored.
        """
