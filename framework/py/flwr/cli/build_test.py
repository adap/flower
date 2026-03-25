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

    fab, _ = build_fab_from_files(files)

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


def test_build_fab_from_files_defaults_fab_format_version() -> None:
    """Test missing fab_format_version defaults in returned metadata."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b'[project]\nname = "app"\nversion = "1.0.0"\n',
        "client.py": b"print('ok')\n",
    }

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None


def test_build_fab_from_files_preserves_target_for_version_zero() -> None:
    """Test fab_format_version=0 accepts flwr_version_target without bounds."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"

[tool.flwr.app]
publisher = "alice"
fab_format_version = 0
flwr_version_target = "1.27.1"
""",
        "client.py": b"print('ok')\n",
    }

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target == "1.27.1"
    assert metadata.flwr_version_max is None


def test_build_fab_from_files_derives_flwr_bounds() -> None:
    """Test fab_format_version=1 derives metadata."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"
dependencies = ["flwr[simulation]>=1.26.0,<=1.28.0", "numpy>=1.0.0"]

[tool.flwr.app]
publisher = "alice"
fab_format_version = 1
flwr_version_target = "1.27.1"
""",
        "client.py": b"print('ok')\n",
    }

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target == "1.27.1"
    assert metadata.flwr_version_max == "1.28.0"


def test_build_fab_from_files_rejects_unsupported_fab_format_version() -> None:
    """Test build fails for unsupported fab_format_version values."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"

[tool.flwr.app]
publisher = "alice"
fab_format_version = 2
""",
        "client.py": b"print('ok')\n",
    }

    with pytest.raises(ValueError, match="Unsupported"):
        build_fab_from_files(files)


def test_build_fab_from_files_skips_unsupported_bounds_for_version_zero() -> None:
    """Test fab_format_version=0 keeps target metadata without derivation fallback."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"
dependencies = ["flwr>1.26.0"]

[tool.flwr.app]
publisher = "alice"
fab_format_version = 0
flwr_version_target = "1.27.1"
""",
        "client.py": b"print('ok')\n",
    }

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target == "1.27.1"
    assert metadata.flwr_version_max is None


def test_build_fab_from_files_rejects_unsupported_flwr_specifier() -> None:
    """Test build fails for fab_format_version=1 with an exclusive lower bound."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"
dependencies = ["flwr>1.26.0"]

[tool.flwr.app]
publisher = "alice"
fab_format_version = 1
flwr_version_target = "1.27.1"
""",
        "client.py": b"print('ok')\n",
    }

    with pytest.raises(ValueError, match="inclusive lower bound"):
        build_fab_from_files(files)


def test_build_fab_from_files_requires_target_for_version_one() -> None:
    """Test build fails for fab_format_version=1 without flwr_version_target."""
    files: dict[str, bytes | Path] = {
        "pyproject.toml": b"""
[project]
name = "app"
version = "1.0.0"
dependencies = ["flwr>=1.26.0,<=1.28.0"]

[tool.flwr.app]
publisher = "alice"
fab_format_version = 1
""",
        "client.py": b"print('ok')\n",
    }

    with pytest.raises(ValueError, match="flwr_version_target"):
        build_fab_from_files(files)
