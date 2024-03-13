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
"""Test for Flower command line interface `run` command."""

import os
import textwrap
from typing import Any, Dict

from .flower_toml import load, validate, validate_fields


def test_load_flower_toml_load_from_cwd(tmp_path: str) -> None:
    """Test if load_template returns a string."""
    # Prepare
    flower_toml_content = """
        [project]
        name = "fedgpt"

        [flower.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [flower.engine]
        name = "simulation" # optional

        [flower.engine.simulation.supernode]
        count = 10 # optional
    """
    expected_config = {
        "project": {
            "name": "fedgpt",
        },
        "flower": {
            "components": {
                "serverapp": "fedgpt.server:app",
                "clientapp": "fedgpt.client:app",
            },
            "engine": {
                "name": "simulation",
                "simulation": {"supernode": {"count": 10}},
            },
        },
    }

    # Current directory
    origin = os.getcwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open("flower.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(flower_toml_content))

        # Execute
        config = load()

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_load_flower_toml_from_path(tmp_path: str) -> None:
    """Test if load_template returns a string."""
    # Prepare
    flower_toml_content = """
        [project]
        name = "fedgpt"

        [flower.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [flower.engine]
        name = "simulation" # optional

        [flower.engine.simulation.supernode]
        count = 10 # optional
    """
    expected_config = {
        "project": {
            "name": "fedgpt",
        },
        "flower": {
            "components": {
                "serverapp": "fedgpt.server:app",
                "clientapp": "fedgpt.client:app",
            },
            "engine": {
                "name": "simulation",
                "simulation": {"supernode": {"count": 10}},
            },
        },
    }

    # Current directory
    origin = os.getcwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open("flower.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(flower_toml_content))

        # Execute
        config = load(path=os.path.join(tmp_path, "flower.toml"))

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_validate_flower_toml_fields_empty() -> None:
    """Test that validate_flower_toml_fields fails correctly."""
    # Prepare
    config: Dict[str, Any] = {}

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert not is_valid
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_flower_toml_fields_no_flower() -> None:
    """Test that validate_flower_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        }
    }

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0


def test_validate_flower_toml_fields_no_flower_components() -> None:
    """Test that validate_flower_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "flower": {},
    }

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0


def test_validate_flower_toml_fields_no_server_and_client_app() -> None:
    """Test that validate_flower_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "flower": {"components": {}},
    }

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert not is_valid
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_flower_toml_fields() -> None:
    """Test that validate_flower_toml_fields succeeds correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "flower": {"components": {"serverapp": "", "clientapp": ""}},
    }

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert is_valid
    assert len(errors) == 0
    assert len(warnings) == 0


def test_validate_flower_toml() -> None:
    """Test that validate_flower_toml succeeds correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "flower": {
            "components": {
                "serverapp": "flwr.cli.run:run",
                "clientapp": "flwr.cli.run:run",
            }
        },
    }

    # Execute
    is_valid, errors, warnings = validate(config)

    # Assert
    assert is_valid
    assert not errors
    assert not warnings


def test_validate_flower_toml_fail() -> None:
    """Test that validate_flower_toml fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "flower": {
            "components": {
                "serverapp": "flwr.cli.run:run",
                "clientapp": "flwr.cli.run:runa",
            }
        },
    }

    # Execute
    is_valid, errors, warnings = validate(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0
