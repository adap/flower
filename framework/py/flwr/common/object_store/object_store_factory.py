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
"""Factory class that creates ObjectStore instances."""


from logging import DEBUG
from typing import Optional

from flwr.common.logger import log

from .in_memory_object_store import InMemoryObjectStore
from .object_store import ObjectStore


class ObjectStoreFactory:
    """Factory class that creates ObjectStore instances."""

    def __init__(self) -> None:
        self.store_instance: Optional[ObjectStore] = None

    def store(self) -> ObjectStore:
        """Return an ObjectStore instance and create it, if necessary.

        Returns
        -------
        ObjectStore
            An ObjectStore instance for storing objects by object_id.
        """
        if self.store_instance is None:
            self.store_instance = InMemoryObjectStore()
        log(DEBUG, "Using InMemoryObjectStore")
        return self.store_instance
