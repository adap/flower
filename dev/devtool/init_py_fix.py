# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Fix provided directory and sub-directories for unsorted __all__ in __init__.py files.

Example:
    python -m devtool.init_py_fix framework/py/flwr
"""


import os
import sys
<<<<<<<< HEAD:dev/flwr_dev/init_py_fix.py
from typing import Annotated, List
========
>>>>>>>> 5b3b887a46966820460795e7bc1b7849ee2c023c:dev/devtool/init_py_fix.py

import black
import typer

<<<<<<<< HEAD:dev/flwr_dev/init_py_fix.py
from flwr_dev.init_py_check import get_all_var_list, get_init_dir_list_and_warnings
========
from devtool.init_py_check import get_all_var_list, get_init_dir_list_and_warnings
>>>>>>>> 5b3b887a46966820460795e7bc1b7849ee2c023c:dev/devtool/init_py_fix.py


def fix_all_init_files(dir_list: list[str]) -> None:
    """Sort the __all__ variables that are in __init__.py files."""
    warning_list = []

    for init_dir in dir_list:
        init_file, all_list, all_lines = get_all_var_list(init_dir)

        if all_list:
            sorted_all_list = sorted(all_list)
            if not all_list == sorted_all_list:
                warning_message = "- " + str(init_dir)
                warning_list.append(warning_message)

                old_all_lines = "\n".join(all_lines)
                new_all_lines = (
                    old_all_lines.split("=", 1)[0]
                    + "= "
                    + str(sorted_all_list)[:-1]
                    + ",]"
                )

                new_content = init_file.read_text().replace(
                    old_all_lines, new_all_lines
                )

                # Write the fixed content back to the file
                init_file.write_text(new_content)

                # Format the file with black
                black.format_file_in_place(
                    init_file,
                    fast=False,
                    mode=black.FileMode(),
                    write_back=black.WriteBack.YES,
                )

    if len(warning_list) > 0:
        print("'__all__' lists in the following '__init__.py' files have been sorted:")
        for warning in warning_list:
            print(warning)


def fix_init(
    paths: Annotated[list[str], typer.Argument(help="Path of the files to analyze")]
):
    """Sort variables inside __init__.py files."""
    for path in paths:
        abs_path: str = os.path.abspath(os.path.join(os.getcwd(), path))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        fix_all_init_files(init_dirs)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception(  # pylint: disable=W0719
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    fix_init(sys.argv)
