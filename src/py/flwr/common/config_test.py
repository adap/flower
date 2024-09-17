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
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from flwr.common.typing import UserConfig

from .config import (
    flatten_dict,
    fuse_dicts,
    get_flwr_dir,
    get_project_config,
    get_project_dir,
    parse_config_args,
    unflatten_dict,
)

# Mock constants
FAB_CONFIG_FILE = "pyproject.toml"


def test_get_flwr_dir_with_provided_path() -> None:
    """Test get_flwr_dir with a provided valid path."""
    provided_path = "."
    assert get_flwr_dir(provided_path) == Path(provided_path).absolute()


def test_get_flwr_dir_without_provided_path() -> None:
    """Test get_flwr_dir without a provided path, using default home directory."""
    with patch.dict(os.environ, {"HOME": "/home/user"}):
        assert get_flwr_dir() == Path("/home/user/.flwr")


def test_get_flwr_dir_with_flwr_home() -> None:
    """Test get_flwr_dir with FLWR_HOME environment variable set."""
    with patch.dict(os.environ, {"FLWR_HOME": "/custom/flwr/home"}):
        assert get_flwr_dir() == Path("/custom/flwr/home")


def test_get_flwr_dir_with_xdg_data_home() -> None:
    """Test get_flwr_dir with FLWR_HOME environment variable set."""
    with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/data/home"}):
        assert get_flwr_dir() == Path("/custom/data/home/.flwr")


def test_get_project_dir_invalid_fab_id() -> None:
    """Test get_project_dir with an invalid fab_id."""
    with pytest.raises(ValueError):
        get_project_dir("invalid_fab_id", "1.0.0")


def test_get_project_dir_valid() -> None:
    """Test get_project_dir with an valid fab_id and version."""
    app_path = get_project_dir("app_name/user", "1.0.0", flwr_dir=".")
    assert app_path == Path("apps") / "app_name" / "user" / "1.0.0"


def test_get_project_config_file_not_found() -> None:
    """Test get_project_config when the configuration file is not found."""
    with pytest.raises(FileNotFoundError):
        get_project_config("/invalid/dir")


def test_get_fused_config_valid(tmp_path: Path) -> None:
    """Test get_project_config when the configuration file is not found."""
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.9.0,<2.0",
            "numpy>=1.21.0",
        ]

        [tool.flwr.app]
        publisher = "flwrlabs"

        [tool.flwr.app.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [tool.flwr.app.config]
        num_server_rounds = 10
        momentum = 0.1
        lr = 0.01
        progress_bar = true
        serverapp.test = "key"

        [tool.flwr.app.config.clientapp]
        test = "key"
    """
    overrides: UserConfig = {
        "num_server_rounds": 5,
        "lr": 0.2,
        "serverapp.test": "overriden",
    }
    expected_config = {
        "num_server_rounds": 5,
        "momentum": 0.1,
        "lr": 0.2,
        "progress_bar": True,
        "serverapp.test": "overriden",
        "clientapp.test": "key",
    }
    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open(FAB_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        default_config = get_project_config(tmp_path)["tool"]["flwr"]["app"].get(
            "config", {}
        )

        config = fuse_dicts(flatten_dict(default_config), overrides)

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_get_project_config_file_valid(tmp_path: Path) -> None:
    """Test get_project_config when the configuration file is not found."""
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.9.0,<2.0",
            "numpy>=1.21.0",
        ]

        [tool.flwr.app]
        publisher = "flwrlabs"

        [tool.flwr.app.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [tool.flwr.app.config]
        num_server_rounds = 10
        momentum = 0.1
        progress_bar = true
        lr = "0.01"
    """
    expected_config = {
        "build-system": {"build-backend": "hatchling.build", "requires": ["hatchling"]},
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"text": "Apache License (2.0)"},
            "dependencies": ["flwr[simulation]>=1.9.0,<2.0", "numpy>=1.21.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "fedgpt.server:app",
                        "clientapp": "fedgpt.client:app",
                    },
                    "config": {
                        "num_server_rounds": 10,
                        "momentum": 0.1,
                        "progress_bar": True,
                        "lr": "0.01",
                    },
                },
            },
        },
    }
    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open(FAB_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = get_project_config(tmp_path)

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_flatten_dict() -> None:
    """Test flatten_dict with a nested dictionary."""
    raw_dict = {"a": {"b": {"c": "d"}}, "e": "f"}
    expected = {"a.b.c": "d", "e": "f"}
    assert flatten_dict(raw_dict) == expected


def test_unflatten_dict() -> None:
    """Test unflatten_dict with a flat dictionary."""
    raw_dict = {"a.b.c": "d", "e": "f"}
    expected = {"a": {"b": {"c": "d"}}, "e": "f"}
    assert unflatten_dict(raw_dict) == expected


def test_parse_config_args_none() -> None:
    """Test parse_config_args with None as input."""
    assert not parse_config_args(None)


def test_parse_config_args_overrides() -> None:
    """Test parse_config_args with key-value pairs."""
    assert parse_config_args(
        ["key1='value1' key2='value2'", "key3=1", "key4=2.0 key5=true key6='value6'"]
    ) == {
        "key1": "value1",
        "key2": "value2",
        "key3": 1,
        "key4": 2.0,
        "key5": True,
        "key6": "value6",
    }


def test_parse_config_args_from_toml_file() -> None:
    """Test if a toml passed to --run-config it is loaded and fused correctly."""
    # Will be saved as a temp .toml file
    toml_config = """
        num-server-rounds = 10
        momentum = 0.1
        verbose = true
    """
    # This is the UserConfig that would be extracted from pyproject.toml
    initial_run_config: UserConfig = {
        "num-server-rounds": 5,
        "momentum": 0.2,
        "dataset": "my-fancy-dataset",
        "verbose": False,
    }
    expected_config = {
        "num-server-rounds": 10,
        "momentum": 0.1,
        "dataset": "my-fancy-dataset",
        "verbose": True,
    }

    # Create a temporary directory using a context manager
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file within that directory
        toml_config_file = os.path.join(temp_dir, "extra_config.toml")

        # Write the data to the TOML file
        with open(toml_config_file, "w", encoding="utf-8") as toml_file:
            toml_file.write(textwrap.dedent(toml_config))

        # Parse config (this mimics what `--run-config path/to/config.toml` does)
        config_from_toml = parse_config_args([toml_config_file])
        # Fuse
        config = fuse_dicts(initial_run_config, config_from_toml)

        # Assert
        assert config == expected_config


def test_parse_config_args_passing_toml_and_key_value() -> None:
    """Test that passing a toml and key-value configs aren't allowed."""
    config = ["my-other-config.toml", "lr=0.1", "epochs=99"]
    with pytest.raises(ValueError):
        parse_config_args(config)
