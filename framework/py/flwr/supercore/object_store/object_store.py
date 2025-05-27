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
    def preregister(self, object_ids: list[str]) -> list[str]:
        """Preregister objects in the `ObjectStore`.

        Parameters
        ----------
        object_ids : list[str]
            The list of object_ids to be pre-registered in the store if they do not
            exist.

        Returns
        -------
        list[str]
            List of object_ids that were preregistered. This list represents the
            object_ids that were not present in the `ObjectStore` at the time this
            method was executed.
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
    def set_children_object_ids(
        self, msg_object_id: str, children_ids: list[str]
    ) -> None:
        """Store mapping of an object_id of type ``Message`` to those of its children.

        Parameters
        ----------
        msg_object_id : str
            The object_id of a ``Message``.

        children_ids : list[str]
            A list of object_ids belonging to the children of the ``Message``.
        """

    @abc.abstractmethod
    def get_children_object_ids(self, msg_object_id: str) -> list[str]:
        """Get object_ids of childrens.

        Parameters
        ----------
        msg_object_id : str
            The object_id of a ``Message`` object.

        Returns
        -------
        list[str]
            The list of object_ids of children objects.
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
