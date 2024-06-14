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
"""Provide functions for managing global Flower config."""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

APP_DIR = "apps"


def get_flwr_dir() -> Path:
    """Return the Flower home directory based on env variables."""
    return Path(
        os.getenv(
            "FLWR_HOME",
            f"{os.getenv('XDG_DATA_HOME', os.getenv('HOME'))}/.flwr",
        )
    )


def get_project_dir(
    fab_id: str, fab_version: str, flwr_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Return the project directory based on the given fab_id, fab_version."""
    if flwr_dir is None:
        flwr_dir = get_flwr_dir()

    publisher, project_name = fab_id.split("/")
    return Path(flwr_dir) / APP_DIR / publisher / project_name / fab_version


def get_project_config(project_dir: Union[str, Path]) -> Dict[str, Any]:
    """"""