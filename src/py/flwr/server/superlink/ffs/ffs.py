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
from typing import Dict, List, Tuple


class Ffs(abc.ABC):  # pylint: disable=R0904
    """Abstract Flower File Storage interface for large objects."""

    @abc.abstractmethod
    def put(self, content: bytes, meta: Dict[str, str]) -> str:
        """Store bytes and metadata and return sha256hex hash of data as str.

        Parameters
        ----------
        content : bytes
            The content to be stored.
        meta : Dict[str, str]
            The metadata to be stored.

        Returns
        -------
        sha256hex : string
            The sha256hex hash of the content.
        """

    @abc.abstractmethod
    def get(self, key: str) -> Tuple[bytes, Dict[str, str]]:
        """Return tuple containing the object and it's meta fields.

        Parameters
        ----------
        hash : string
            The sha256hex hash of the object to be retrieved.

        Returns
        -------
        Tuple[bytes, Dict[str, str]]
            A tuple containing the object and it's metadata.
        """

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete object with hash.

        Parameters
        ----------
        hash : string
            The sha256hex hash of the object to be deleted.
        """

    @abc.abstractmethod
    def list(self) -> List[str]:
        """List all keys.

        Can be used to list all keys in the storage and e.g. to clean up
        the storage with the delete method.

        Returns
        -------
        List[str]
            A list of all keys.
        """
