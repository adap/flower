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
"""Test for Flower command line interface `new` command."""

import os
from pathlib import Path

import pytest

from .new import MlFramework, create_file, load_template, new, render_template


def test_load_template() -> None:
    """Test if load_template returns a string."""
    # Prepare
    filename = "app/README.md.tpl"

    # Execute
    text = load_template(filename)

    # Assert
    assert isinstance(text, str)


def test_render_template() -> None:
    """Test if a string is correctly substituted."""
    # Prepare
    filename = "app/README.md.tpl"
    data = {
        "framework_str": "",
        "project_name": "FedGPT",
        "package_name": "fedgpt",
        "import_name": "fedgpt",
        "username": "flwrlabs",
    }

    # Execute
    result = render_template(filename, data)

    # Assert
    assert "# FedGPT" in result


def test_create_file(tmp_path: str) -> None:
    """Test if file with content is created."""
    # Prepare
    file_path = Path(tmp_path) / "test.txt"
    content = "Foobar"

    # Execute
    create_file(file_path, content)

    # Assert
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    assert text == "Foobar"


def test_new_correct_name(tmp_path: str) -> None:
    """Test if project with correct name is created for framework."""
    # Prepare
    framework = MlFramework.PYTORCH
    username = "flwrlabs"
    expected_names = [
        ("FedGPT", "fedgpt", "fedgpt"),
        ("My-Flower-App", "my-flower-app", "my_flower_app"),
    ]

    for project_name, expected_top_level_dir, expected_module_dir in expected_names:
        expected_files_top_level = {
            expected_module_dir,
            "README.md",
            "pyproject.toml",
            ".gitignore",
        }
        expected_files_module = {
            "__init__.py",
            "server_app.py",
            "client_app.py",
            "task.py",
        }

        # Current directory
        origin = Path.cwd()

        try:
            # Change into the temprorary directory
            os.chdir(tmp_path)
            # Execute
            new(project_name=project_name, framework=framework, username=username)

            # Assert
            file_list = (Path(tmp_path) / expected_top_level_dir).iterdir()
            assert {
                file_path.name for file_path in file_list
            } == expected_files_top_level

            file_list = (
                Path(tmp_path) / expected_top_level_dir / expected_module_dir
            ).iterdir()
            assert {file_path.name for file_path in file_list} == expected_files_module
        finally:
            os.chdir(origin)


def test_new_incorrect_name(tmp_path: str) -> None:
    """Test if project with incorrect name is created for framework."""
    framework = MlFramework.PYTORCH
    username = "flwrlabs"

    for project_name in ["My_Flower_App", "My.Flower App"]:
        # Current directory
        origin = Path.cwd()

        try:
            # Change into the temprorary directory
            os.chdir(tmp_path)

            with pytest.raises(OSError) as exc_info:

                # Execute
                new(
                    project_name=project_name,
                    framework=framework,
                    username=username,
                )

                assert "Failed to read from stdin" in str(exc_info.value)

        finally:
            os.chdir(origin)
