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
from pathlib import Path
from typing import Any, Dict

from .config_utils import load, validate, validate_fields


def test_load_pyproject_toml_load_from_cwd(tmp_path: Path) -> None:
    """Test if load_template returns a string."""
    # Prepare
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        authors = [
            { name = "The Flower Authors", email = "hello@flower.ai" },
        ]
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.8.0,<2.0",
            "numpy>=1.21.0",
        ]

        [flower]
        publisher = "flwrlabs"

        [flower.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [flower.engine]
        name = "simulation" # optional

        [flower.engine.simulation.supernode]
        count = 10 # optional
    """
    expected_config = {
        "build-system": {"build-backend": "hatchling.build", "requires": ["hatchling"]},
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "authors": [{"email": "hello@flower.ai", "name": "The Flower Authors"}],
            "license": {"text": "Apache License (2.0)"},
            "dependencies": ["flwr[simulation]>=1.8.0,<2.0", "numpy>=1.21.0"],
        },
        "flower": {
            "publisher": "flwrlabs",
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
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open("pyproject.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = load()

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_load_pyproject_toml_from_path(tmp_path: Path) -> None:
    """Test if load_template returns a string."""
    # Prepare
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        authors = [
            { name = "The Flower Authors", email = "hello@flower.ai" },
        ]
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.8.0,<2.0",
            "numpy>=1.21.0",
        ]

        [flower]
        publisher = "flwrlabs"

        [flower.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"

        [flower.engine]
        name = "simulation" # optional

        [flower.engine.simulation.supernode]
        count = 10 # optional
    """
    expected_config = {
        "build-system": {"build-backend": "hatchling.build", "requires": ["hatchling"]},
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "authors": [{"email": "hello@flower.ai", "name": "The Flower Authors"}],
            "license": {"text": "Apache License (2.0)"},
            "dependencies": ["flwr[simulation]>=1.8.0,<2.0", "numpy>=1.21.0"],
        },
        "flower": {
            "publisher": "flwrlabs",
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
        with open("pyproject.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = load(path=tmp_path / "pyproject.toml")

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_validate_pyproject_toml_fields_empty() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config: Dict[str, Any] = {}

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert not is_valid
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields_no_flower() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
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


def test_validate_pyproject_toml_fields_no_flower_components() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
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
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields_no_server_and_client_app() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
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
    assert len(errors) == 3
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields() -> None:
    """Test that validate_pyproject_toml_fields succeeds correctly."""
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
            "publisher": "flwrlabs",
            "components": {"serverapp": "", "clientapp": ""},
        },
    }

    # Execute
    is_valid, errors, warnings = validate_fields(config)

    # Assert
    assert is_valid
    assert len(errors) == 0
    assert len(warnings) == 0


def test_validate_pyproject_toml() -> None:
    """Test that validate_pyproject_toml succeeds correctly."""
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
            "publisher": "flwrlabs",
            "components": {
                "serverapp": "flwr.cli.run:run",
                "clientapp": "flwr.cli.run:run",
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate(config)

    # Assert
    assert is_valid
    assert not errors
    assert not warnings


def test_validate_pyproject_toml_fail() -> None:
    """Test that validate_pyproject_toml fails correctly."""
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
            "publisher": "flwrlabs",
            "components": {
                "serverapp": "flwr.cli.run:run",
                "clientapp": "flwr.cli.run:runa",
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0
