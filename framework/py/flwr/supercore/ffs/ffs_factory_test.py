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
"""Test Ffs factory."""


from .disk_ffs import DiskFfs
from .ffs_factory import FfsFactory


def test_disk_ffs_factory() -> None:
    """Test DiskFfs instantiation with FfsFactory."""
    # Prepare
    ffs_factory = FfsFactory("test")

    # Execute
    ffs = ffs_factory.ffs()

    # Assert
    assert isinstance(ffs, DiskFfs)


def test_cache_ffs_factory() -> None:
    """Test cache with FfsFactory."""
    # Prepare
    ffs_factory = FfsFactory("other_test")
    ffs = ffs_factory.ffs()

    # Execute
    other_ffs = ffs_factory.ffs()

    # Assert
    assert id(ffs) == id(other_ffs)
