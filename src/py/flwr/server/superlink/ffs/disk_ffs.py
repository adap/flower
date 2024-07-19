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
"""Disk based Flower File Storage."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

from flwr.server.superlink.ffs.ffs import Ffs


class DiskFfs(Ffs):  # pylint: disable=R0904
    """Disk based Flower File Storage interface for large objects."""

    def __init__(self, base_dir: str) -> None:
        """Create a new DiskFfs instance.

        Parameters
        ----------
        base_dir : string
            The base directory to store the objects.
        """
        self.base_dir = Path(base_dir)

    def put(self, content: bytes, meta: Dict[str, str]) -> str:
        """Store bytes and metadata and returns sha256hex hash of data as str.

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
        content_hash = hashlib.sha256(content).hexdigest()

        (self.base_dir / content_hash).write_bytes(content)
        (self.base_dir / f"{content_hash}.META").write_text(json.dumps(meta))

        return content_hash

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
        content = (self.base_dir / key).read_bytes()
        meta = json.loads((self.base_dir / f"{key}.META").read_text())

        return content, meta

    def delete(self, key: str) -> None:
        """Delete object with hash.

        Parameters
        ----------
        hash : string
            The sha256hex hash of the object to be deleted.
        """
        (self.base_dir / key).unlink()
        (self.base_dir / f"{key}.META").unlink()

    def list(self) -> List[str]:
        """List all keys.

        Can be used to list all keys in the storage and e.g. to clean up
        the storage with the delete method.

        Returns
        -------
        List[str]
            A list of all keys.
        """
        return [
            item.name for item in self.base_dir.iterdir() if not item.suffix == ".META"
        ]
