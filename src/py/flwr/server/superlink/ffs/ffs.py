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
"""Abstract base class for Flower File Storage interface."""


import abc
from typing import Optional


class Ffs(abc.ABC):  # pylint: disable=R0904
    """Abstract Flower File Storage interface for large objects."""

    @abc.abstractmethod
    def put(self, content: bytes, meta: dict[str, str]) -> str:
        """Store bytes and metadata and return sha256hex hash of data as str.

        Parameters
        ----------
        content : bytes
            The content to be stored.
        meta : Dict[str, str]
            The metadata to be stored.

        Returns
        -------
        key : str
            The key (sha256hex hash) of the content.
        """

    @abc.abstractmethod
    def get(self, key: str) -> Optional[tuple[bytes, dict[str, str]]]:
        """Return tuple containing the object content and metadata.

        Parameters
        ----------
        key : str
            The key (sha256hex hash) of the object to be retrieved.

        Returns
        -------
        Optional[Tuple[bytes, Dict[str, str]]]
            A tuple containing the object content and metadata.
        """

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete object with hash.

        Parameters
        ----------
        key : str
            The key (sha256hex hash) of the object to be deleted.
        """

    @abc.abstractmethod
    def list(self) -> list[str]:
        """List keys of all stored objects.

        Return all available keys in this `Ffs` instance.
        This can be combined with, for example,
        the `delete` method to delete objects.

        Returns
        -------
        List[str]
            A list of all available keys.
        """
