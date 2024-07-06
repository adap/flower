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
"""Test util functions handling Flower config."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from .config import (
    flatten_dict,
    get_flwr_dir,
    get_project_config,
    get_project_dir,
    parse_config_args,
)

# Mock constants
APP_DIR = "app_dir"
FAB_CONFIG_FILE = "pyproject.toml"
FLWR_HOME = "FLWR_HOME"


def test_get_flwr_dir_with_provided_path() -> None:
    """Test get_flwr_dir with a provided valid path."""
    provided_path = "/some/path"
    assert get_flwr_dir(provided_path) == Path(provided_path).absolute()


def test_get_flwr_dir_without_provided_path() -> None:
    """Test get_flwr_dir without a provided path, using default home directory."""
    with patch.dict(os.environ, {"HOME": "/home/user"}):
        assert get_flwr_dir() == Path("/home/user/.flwr")


def test_get_flwr_dir_with_flwr_home() -> None:
    """Test get_flwr_dir with FLWR_HOME environment variable set."""
    with patch.dict(os.environ, {FLWR_HOME: "/custom/flwr/home"}):
        assert get_flwr_dir() == Path("/custom/flwr/home")


def test_get_project_dir_valid() -> None:
    """Test get_project_dir with valid fab_id and fab_version."""
    with patch("config.get_flwr_dir", return_value=Path("/flwr/home")):
        assert get_project_dir("publisher/project", "1.0.0") == Path(
            "/flwr/home/app_dir/publisher/project/1.0.0"
        )


def test_get_project_dir_invalid_fab_id() -> None:
    """Test get_project_dir with an invalid fab_id."""
    with pytest.raises(ValueError):
        get_project_dir("invalid_fab_id", "1.0.0")


def test_get_project_config_file_not_found() -> None:
    """Test get_project_config when the configuration file is not found."""
    with pytest.raises(FileNotFoundError):
        get_project_config("/invalid/dir")


def test_flatten_dict() -> None:
    """Test flatten_dict with a nested dictionary."""
    raw_dict = {"a": {"b": {"c": "d"}}, "e": "f"}
    expected = {"a.b.c": "d", "e": "f"}
    assert flatten_dict(raw_dict) == expected


def test_parse_config_args_none() -> None:
    """Test parse_config_args with None as input."""
    assert parse_config_args(None) == {}


def test_parse_config_args_overrides() -> None:
    """Test parse_config_args with key-value pairs."""
    assert parse_config_args("key1=value1,key2=value2") == {
        "key1": "value1",
        "key2": "value2",
    }
