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
"""Flower command line interface `example` command."""

import json
import os
import subprocess
import tempfile
import urllib.request

from .utils import prompt_options


def example() -> None:
    """Clone a Flower example.

    All examples available in the Flower repository are available through this command.
    """
    # Load list of examples directly from GitHub
    url = "https://api.github.com/repos/adap/flower/git/trees/main"
    with urllib.request.urlopen(url) as res:
        data = json.load(res)
        examples_directory_url = [
            item["url"] for item in data["tree"] if item["path"] == "examples"
        ][0]

    with urllib.request.urlopen(examples_directory_url) as res:
        data = json.load(res)
        example_names = [
            item["path"]
            for item in data["tree"]
            if item["path"] not in [".gitignore", "doc"]
        ]

    example_name = prompt_options(
        "Please select example by typing in the number",
        example_names,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.check_output(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/adap/flower.git",
                tmpdirname,
            ]
        )
        examples_dir = os.path.join(tmpdirname, "examples", example_name)
        subprocess.check_output(["mv", examples_dir, "."])

        print()
        print(f"Example ready to use in {os.path.join(os.getcwd(), example_name)}")
