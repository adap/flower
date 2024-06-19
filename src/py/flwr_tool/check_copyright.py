# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
"""Check if copyright notices are present in all Python files.

Example:
    python -m flwr_tool.check_copyright src/py/flwr
"""


import os
import subprocess
import sys
from pathlib import Path
from typing import List

from flwr_tool.init_py_check import get_init_dir_list_and_warnings

COPYRIGHT_FORMAT = """# Copyright {} Flower Labs GmbH. All Rights Reserved.
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
# =============================================================================="""


def _get_file_creation_year(filepath: str):
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ai", "--", filepath],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    date_str = result.stdout.splitlines()[-1]  # Get the first commit date
    creation_year = date_str.split("-")[0]  # Extract the year
    return creation_year


def _check_copyright(dir_list: List[str]) -> None:
    warning_list = []
    for valid_dir in dir_list:
        if "proto" in valid_dir:
            continue

        dir_path = Path(valid_dir)
        for py_file in dir_path.glob("*.py"):
            creation_year = _get_file_creation_year(str(py_file.absolute()))
            expected_copyright = COPYRIGHT_FORMAT.format(creation_year)

            if expected_copyright not in py_file.read_text():
                warning_message = "- " + str(py_file)
                warning_list.append(warning_message)

    if len(warning_list) > 0:
        print("Missing or incorrect copyright notice in the following files:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception(  # pylint: disable=W0719
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    for i, _ in enumerate(sys.argv):
        abs_path: str = os.path.abspath(os.path.join(os.getcwd(), sys.argv[i]))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        _check_copyright(init_dirs)
