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

import json
import subprocess
import urllib.request

import typer


def example():
    """
    This command allows you to copy any Flower example which is currently available at:
    https://github.com/adap/flower/tree/main/examples
    """

    # Load list of examples directly from GitHub
    url = "https://api.github.com/repos/adap/flower/git/trees/main"
    res = json.load(urllib.request.urlopen(url))
    examples_directory_url = [
        item["url"] for item in res["tree"] if item["path"] == "examples"
    ][0]
    result = json.load(urllib.request.urlopen(examples_directory_url))
    example_names = [
        item["path"] for item in result["tree"] if item["path"] not in [".gitignore"]
    ]

    # Turn examples into a list with index as in "quickstart-pytorch [0]"
    content = [f" {name} [{index}]" for index, name in enumerate(example_names)]
    print()
    index = typer.prompt(
        "💬 Please select the example you'd like to copy by typing in the number.\n\n"
        + "\n".join(content)
    )
    print()

    example_name = example_names[int(index)]

    subprocess.check_output(
        ["git", "clone", "--depth=1", "https://github.com/adap/flower.git"]
    )
    subprocess.check_output(["mv", f"flower/examples/{example_name}", "."])
    subprocess.check_output(["rm", "-rf", "flower"])

    print()
    print(f"Example ready to use: {example_name}")
