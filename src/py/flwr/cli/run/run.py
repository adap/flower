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
from typing import Dict, Optional, Union
import tomli

import typer

from ..utils import prompt_options


def load_flower_toml() -> Optional[Dict[str, Union[str, int]]]:
    """Load flower.toml and return as dict."""
    cur_dir = os.getcwd()
    toml_path = os.path.join(cur_dir, "flower.toml")

    if not os.path.isfile(toml_path):
        return None

    with open(toml_path, encoding="utf-8") as toml_file:
        data = tomli.loads(toml_file.read())
        return data


def run() -> None:
    """Run Flower project."""

    config = load_flower_toml()

    if not config:
        print(
            typer.style(
                "Project configuration could not be loaded. A valid flower.toml file is required.",
                fg=typer.colors.RED,
                bold=True,
            )
        )

    print(config)
