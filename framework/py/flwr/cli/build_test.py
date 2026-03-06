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
"""Test for Flower command line interface `build` command."""


from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from .build import build_fab_from_files


def test_build_fab_from_files_includes_root_license_and_pyproject() -> None:
    """Test root LICENSE and pyproject.toml are included in FAB when present."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b'[project]\nname = "app"\nversion = "1.0.0"\n',
        "client.py": b"print('ok')\n",
        "LICENSE": b"Apache-2.0\n",
    }

    fab = build_fab_from_files(files)

    with ZipFile(BytesIO(fab), "r") as zip_file:
        names = set(zip_file.namelist())
        content_manifest = zip_file.read(".info/CONTENT").decode("utf-8")

    assert "LICENSE" in names
    assert "LICENSE," in content_manifest
    assert "pyproject.toml" in names
    assert "pyproject.toml," in content_manifest


def test_build_fab_from_files_missing_pyproject_raises() -> None:
    """Test that missing pyproject.toml raises ValueError."""
    files: dict[str, bytes | Path] = {
        "client.py": b"print('ok')\n",
    }

    with pytest.raises(ValueError, match="pyproject.toml"):
        build_fab_from_files(files)
