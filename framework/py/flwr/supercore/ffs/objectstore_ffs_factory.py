# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Factory class that creates ObjectStoreFfs instances."""


from logging import DEBUG

from flwr.common.logger import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME

from .ffs import Ffs
from .ffs_factory import FfsFactory
from .objectstore_ffs import ObjectStoreFfs


class ObjectStoreFfsFactory(FfsFactory):
    """Factory class that creates ObjectStore-backed Ffs instances.

    Parameters
    ----------
    database : str
        The database configuration used by SuperLink state and ObjectStore.
    """

    def __init__(self, database: str = FLWR_IN_MEMORY_DB_NAME) -> None:
        super().__init__(database)
        self.database = database

    def ffs(self) -> Ffs:
        """Return an ObjectStoreFfs instance and create it, if necessary."""
        if self.ffs_instance is None:
            log(DEBUG, "Initializing ObjectStoreFfs")
            self.ffs_instance = ObjectStoreFfs(self.database)

        log(DEBUG, "Using ObjectStoreFfs")
        return self.ffs_instance
