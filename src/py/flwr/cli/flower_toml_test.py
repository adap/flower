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

from .flower_toml import (
    load_flower_toml,
    validate_object_reference,
    validate_flower_toml_fields,
    validate_flower_toml,
)


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

        [flower.engine.simulation.super-node]
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
                "simulation": {"super-node": {"count": 10}},
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
        config = load_flower_toml()

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

        [flower.engine.simulation.super-node]
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
                "simulation": {"super-node": {"count": 10}},
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
        config = load_flower_toml(path=os.path.join(tmp_path, "flower.toml"))

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_validate_flower_toml_fields_empty() -> None:
    # Prepare
    config = {}

    # Execute
    is_valid, reasons = validate_flower_toml_fields(config)

    # Assert
    assert not is_valid
    assert len(reasons) == 2


def test_validate_flower_toml_fields_no_flower() -> None:
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
    is_valid, reasons = validate_flower_toml_fields(config)

    # Assert
    assert not is_valid
    assert len(reasons) == 1


def test_validate_flower_toml_fields_no_flower_components() -> None:
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
    is_valid, reasons = validate_flower_toml_fields(config)

    # Assert
    assert not is_valid
    assert len(reasons) == 1


def test_validate_flower_toml_fields_no_server_and_client_app() -> None:
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
    is_valid, reasons = validate_flower_toml_fields(config)

    # Assert
    assert not is_valid
    assert len(reasons) == 2


def test_validate_flower_toml_fields() -> None:
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
    is_valid, reasons = validate_flower_toml_fields(config)

    # Assert
    assert is_valid
    assert len(reasons) == 0


def test_validate_object_reference() -> None:
    # Prepare
    ref = "flwr.cli.run:run"

    # Execute
    is_valid, error = validate_object_reference(ref)

    # Assert
    assert is_valid
    assert error is None


def test_validate_object_reference_fails() -> None:
    # Prepare
    ref = "flwr.cli.run:runa"

    # Execute
    is_valid, error = validate_object_reference(ref)

    # Assert
    assert not is_valid
    assert error == "Unable to load attribute runa from module flwr.cli.run"


def test_validate_flower_toml() -> None:
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
    is_valid, reasons = validate_flower_toml(config)

    # Assert
    assert is_valid
    assert not reasons


def test_validate_flower_toml_fail() -> None:
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
    is_valid, reasons = validate_flower_toml(config)

    # Assert
    assert not is_valid
    assert len(reasons) == 1
