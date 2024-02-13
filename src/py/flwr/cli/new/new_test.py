# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Test for Flower CLI new command."""

import os

from .new import new, MLFramework, load_template, render_template, create_file


def test_load_template():
    # Prepare
    filename = "README.md"

    # Execute
    text = load_template(filename)

    # Assert
    assert isinstance(text, str)


def test_render_template():
    # Prepare
    filename = "README.md"
    data = {"project_name": "FedGPT"}

    # Execute
    result = render_template(filename, data)

    # Assert
    assert "# FedGPT" in result


def test_create_file(tmp_path):
    # Prepare
    file_path = os.path.join(tmp_path, "test.txt")
    content = "Foobar"

    # Execute
    result = create_file(file_path, content)

    # Assert
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    assert text == "Foobar"


def test_new(tmp_path):
    # Prepare
    project_name = "FedGPT"
    framework = MLFramework.pytorch
    expected_files_top_level = set(
        ["requirements.txt", "fedgpt", "README.md", "flower.toml"]
    )
    expected_files_module = set(
        ["main.py", "__init__.py"]
    )

    ## Change into the temprorary directory
    os.chdir(tmp_path)

    # Execute
    new(project_name=project_name, framework=framework)

    # Assert
    file_list = os.listdir(os.path.join(tmp_path, project_name.lower()))
    assert set(file_list) == expected_files_top_level

    file_list = os.listdir(os.path.join(tmp_path, project_name.lower(), project_name.lower()))
    assert set(file_list) == expected_files_module
