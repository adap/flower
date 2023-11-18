# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Flower Datasets package version helper.

The code is an exact copy from flwr.
"""


import importlib.metadata as importlib_metadata
from typing import Tuple


def _check_package(name: str) -> Tuple[str, str]:
    version: str = importlib_metadata.version(name)
    return name, version


def _version() -> Tuple[str, str]:
    """Read and return Flower Dataset package name and version.

    Returns
    -------
    package_name, package_version : Tuple[str, str]
    """
    for name in ["flwr-datasets", "flwr-datasets-nightly"]:
        try:
            return _check_package(name)
        except importlib_metadata.PackageNotFoundError:
            pass

    return ("unknown", "unknown")


package_name, package_version = _version()
