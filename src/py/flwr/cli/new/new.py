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
"""Flower CLI package."""

from typing import Dict
from string import Template
from enum import Enum
import os

import typer
from typing_extensions import Annotated


class MLFramework(str, Enum):
    pytorch = "PyTorch"
    # tensorflow = "TensorFlow"


def load_template(name: str):
    """Load template from template directory and return as text."""
    tpl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    tpl_files = os.listdir(tpl_dir)

    if name not in tpl_files:
        raise Exception(f"Template '{name}' not found")

    with open(os.path.join(tpl_dir, name), encoding="utf-8") as tpl_file:
        return tpl_file.read()


def render_template(template_name: str, data: Dict[str, str]):
    """Render template."""
    tpl_file = load_template(template_name)
    tpl = Template(tpl_file)
    result = tpl.substitute(data)
    return result


def create_file(file_path: str, content: str):
    """Create file including all nessecary directories and write content into file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def render_and_create(file_path: str, template_name: str, data: Dict[str, str]):
    """Render template and write to file."""
    content = render_template(template_name, data)
    create_file(file_path, content)


def new(
    project_name: Annotated[
        str,
        typer.Argument(metavar="🌼 Project name 🌼", help="The name of the project"),
    ],
    framework: Annotated[
        MLFramework,
        typer.Option(
            case_sensitive=False,
            help="The ML framework to use",
            prompt=("💬 Please select your machine learning framework."),
        ),
    ],
):
    """
    This command will guide you through creating your Flower project.
    """
    print(f"Creating Flower project {project_name}...")

    # Set project directory path
    cwd = os.getcwd()
    project_dir = os.path.join(cwd, project_name.lower())

    # Render README.md
    file_path = os.path.join(project_dir, "README.md")
    render_and_create(file_path, "README.md", {"project_name": project_name})

    # Render requirements.txt
    file_path = os.path.join(project_dir, "requirements.txt")
    render_and_create(file_path, f"requirements.{framework.lower()}.txt", {})

    # Render flower.toml
    file_path = os.path.join(project_dir, "flower.toml")
    render_and_create(file_path, f"flower.toml", {"project_name": project_name})

    # Render __init__.py in module directory
    file_path = os.path.join(project_dir, f"{project_name.lower()}/__init__.py")
    render_and_create(file_path, f"__init__.py", {})

    # Render main.py in module directory
    file_path = os.path.join(project_dir, f"{project_name.lower()}/main.py")
    render_and_create(file_path, f"main.py", {})

    print(f"Project creation successful.")
