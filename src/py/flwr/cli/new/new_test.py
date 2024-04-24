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
        "project_name": "FedGPT",
        "distribution_name": "fedgpt",
        "import_name": "fedgpt",
    }

    # Execute
    result = render_template(filename, data)

    # Assert
    assert "# FedGPT" in result


def test_create_file(tmp_path: str) -> None:
    """Test if file with content is created."""
    # Prepare
    file_path = os.path.join(tmp_path, "test.txt")
    content = "Foobar"

    # Execute
    create_file(file_path, content)

    # Assert
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    assert text == "Foobar"


def test_new(tmp_path: str) -> None:
    """Test if project is created for framework."""
    # Prepare
    expected_names = [
        ("FedGPT", "FedGPT", "fedgpt"),
        ("My_Flower-App", "My_Flower-App", "my-flower-app"),
    ]

    for project_name, expected_top_level_dir, expected_module_dir in expected_names:
        framework = MlFramework.PYTORCH
        expected_files_top_level = {
            expected_module_dir,
            "README.md",
            "pyproject.toml",
            ".gitignore",
        }
        expected_files_module = {
            "__init__.py",
            "server.py",
            "client.py",
            "task.py",
        }

        # Current directory
        origin = os.getcwd()

        try:
            # Change into the temprorary directory
            os.chdir(tmp_path)

            # Execute
            new(project_name=project_name, framework=framework)

            # Assert
            file_list = os.listdir(os.path.join(tmp_path, expected_top_level_dir))
            assert set(file_list) == expected_files_top_level

            file_list = os.listdir(
                os.path.join(tmp_path, expected_top_level_dir, expected_module_dir)
            )
            assert set(file_list) == expected_files_module
        finally:
            os.chdir(origin)
