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
from typing import Any, Optional

import click
import pytest

from .config_utils import (
    load,
    process_loaded_project_config,
    validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_fields,
)


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
                },
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
        config = load(toml_path=Path.cwd() / "pyproject.toml")

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
                },
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
        config = load(toml_path=tmp_path / "pyproject.toml")

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_validate_pyproject_toml_fields_empty() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config: dict[str, Any] = {}

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
        "tool": {"flwr": {"app": {}}},
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
        "tool": {"flwr": {"app": {"components": {}}}},
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
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {"serverapp": "", "clientapp": ""},
                },
            },
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
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
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
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:runa",
                    },
                },
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0


def test_validate_project_config_fail() -> None:
    """Test that validate_project_config fails correctly."""
    # Prepare
    config = None
    errors = ["Error"]
    warnings = ["Warning"]

    # Execute
    with pytest.raises(click.exceptions.Exit) as excinfo:
        _ = process_loaded_project_config(config, errors, warnings)

    # Assert
    assert excinfo.value.exit_code == 1


def test_validate_federation_in_project_config() -> None:
    """Test that validate_federation_in_config succeeds correctly."""
    # Prepare - Test federation is None
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
                "federations": {
                    "default": "default_federation",
                    "default_federation": {"default_key": "default_val"},
                },
            },
        },
    }
    federation = None

    # Execute
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    # Assert
    assert federation == "default_federation"
    assert federation_config == {"default_key": "default_val"}

    federation = "new_federation"
    config["tool"]["flwr"]["federations"]["new_federation"] = {"new_key": "new_val"}

    # Execute
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    # Assert
    assert federation == "new_federation"
    assert federation_config == {"new_key": "new_val"}


def test_validate_federation_in_project_config_fail() -> None:
    """Test that validate_federation_in_config fails correctly."""

    def run_and_assert_exit(federation: Optional[str], config: dict[str, Any]) -> None:
        """Execute validation and assert exit code is 1."""
        with pytest.raises(click.exceptions.Exit) as excinfo:
            validate_federation_in_project_config(federation, config)
        assert excinfo.value.exit_code == 1

    # Prepare
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
                "federations": {},
            },
        },
    }
    federation = None

    # Test federation is None and no default federation is declared
    # Execute and assert
    run_and_assert_exit(federation, config)

    # Prepare - Test federation name is not in config
    federation = "fed_not_in_config"
    config["tool"]["flwr"]["federations"] = {"fed_in_config": {}}

    # Execute and assert
    run_and_assert_exit(federation, config)


def test_validate_certificate_in_federation_config(tmp_path: Path) -> None:
    """Test that validate_certificate_in_federation_config succeeds correctly."""
    # Prepare
    config: dict[str, Any] = {
        "address": "127.0.0.1:9091",
        "root-certificates": "dummy_cert.pem",
    }
    dummy_cert = tmp_path / "dummy_cert.pem"
    dummy_cert.write_text("dummy_cert")

    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)

        # Test insecure is not declared and root_certificates is present
        # Execute
        insecure, root_cert = validate_certificate_in_federation_config(
            tmp_path, config
        )
        # Assert
        assert not insecure
        assert root_cert == b"dummy_cert"

        # Test insecure is False and root_certificates is present
        config["insecure"] = False
        # Execute
        insecure, root_cert = validate_certificate_in_federation_config(
            tmp_path, config
        )
        # Assert
        assert not insecure
        assert root_cert == b"dummy_cert"

        # Test insecure is True and root_certificates is None
        config["insecure"] = True
        config.pop("root-certificates")

        # Execute
        insecure, root_cert = validate_certificate_in_federation_config(
            tmp_path, config
        )
        # Assert
        assert insecure
        assert root_cert is None
    finally:
        os.chdir(origin)


def test_validate_certificate_in_federation_config_fail(tmp_path: Path) -> None:
    """Test that validate_certificate_in_federation_config fails correctly."""

    def run_and_assert_exit(app: Path, config: dict[str, Any]) -> None:
        """Execute validation and assert exit code is 1."""
        with pytest.raises(click.exceptions.Exit) as excinfo:
            validate_certificate_in_federation_config(app, config)
        assert excinfo.value.exit_code == 1

    # Prepare
    config: dict[str, Any] = {"address": "localhost:8080"}
    dummy_cert = tmp_path / "dummy_cert.pem"
    dummy_cert.write_text("dummy_cert")

    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)

        # Test insecure is None and root_certificates is None
        config["insecure"] = None
        # Execute and assert
        run_and_assert_exit(tmp_path, config)

        # Test insecure is False, but root_certificates is None
        config["insecure"] = False
        # Execute and assert
        run_and_assert_exit(tmp_path, config)

        # Test insecure is True, but root_certificates is not None
        config["root-certificates"] = "dummy_cert.pem"
        config["insecure"] = True
        # Execute and assert
        run_and_assert_exit(tmp_path, config)
    finally:
        os.chdir(origin)
