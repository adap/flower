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
"""Flower command line interface `new` command."""

import os
from enum import Enum
from string import Template
from typing import Dict

import typer
from typing_extensions import Annotated

from ..utils import prompt_options


class MLFramework(str, Enum):
    """Available frameworks."""

    pytorch = "PyTorch"
    tensorflow = "TensorFlow"


def load_template(name: str) -> str:
    """Load template from template directory and return as text."""
    tpl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    tpl_files = os.listdir(tpl_dir)

    if name not in tpl_files:
        raise Exception(f"Template '{name}' not found")

    with open(os.path.join(tpl_dir, name), encoding="utf-8") as tpl_file:
        return tpl_file.read()


def render_template(template: str, data: Dict[str, str]) -> str:
    """Render template."""
    tpl_file = load_template(template)
    tpl = Template(tpl_file)
    result = tpl.substitute(data)
    return result


def create_file(file_path: str, content: str) -> None:
    """Create file including all nessecary directories and write content into file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def render_and_create(file_path: str, template: str, context: Dict[str, str]) -> None:
    """Render template and write to file."""
    content = render_template(template, context)
    create_file(file_path, content)


def new(
    project_name: Annotated[
        str,
        typer.Argument(metavar="project_name", help="The name of the project"),
    ],
    framework: Annotated[
        MLFramework,
        typer.Option(case_sensitive=False, help="The ML framework to use"),
    ] = None,
) -> None:
    """Create new Flower project."""
    print(f"Creating Flower project {project_name}...")

    if framework is None:
        framework_value = prompt_options(
            "Please select ML framework by typing in the number",
            [mlf.value for mlf in MLFramework],
        )
        framework = next(
            name
            for name, value in vars(MLFramework).items()
            if value == framework_value
        )

    # Set project directory path
    cwd = os.getcwd()
    pnl = project_name.lower()
    project_dir = os.path.join(cwd, pnl)

    # List of files to render
    files = {
        "README.md": {
            "template": "README.md",
        },
        "requirements.txt": {"template": f"requirements.{framework.lower()}.txt"},
        "flower.toml": {"template": "flower.toml"},
        f"{pnl}/__init__.py": {"template": "__init__.py"},
        f"{pnl}/main.py": {"template": f"main.{framework.lower()}.py"},
    }
    context = {"project_name": project_name}

    for file_path, value in files.items():
        render_and_create(
            file_path=os.path.join(project_dir, file_path),
            template=value["template"],
            context=context,
        )

    print("Project creation successful.")
