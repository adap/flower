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
"""Factory class that creates Ffs instances."""


from logging import DEBUG
from typing import Optional

from flwr.common.logger import log

from .disk_ffs import DiskFfs
from .ffs import Ffs


class FfsFactory:
    """Factory class that creates Ffs instances.

    Parameters
    ----------
    base_dir : str
        The base directory to store the objects.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.ffs_instance: Optional[Ffs] = None

    def ffs(self) -> Ffs:
        """Return a Ffs instance and create it, if necessary."""
        # SqliteState
        ffs = DiskFfs(self.base_dir)
        log(DEBUG, "Using Disk Flower File System")
        return ffs
