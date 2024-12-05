# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
"""Fix provided directory and sub-directories for unsorted __all__ in __init__.py files.

Example:
    python -m flwr_tool.init_py_fix src/py/flwr
"""


import os
import sys
from typing import List

import black

from flwr_tool.init_py_check import get_all_var_list, get_init_dir_list_and_warnings


def fix_all_init_files(dir_list: List[str]) -> None:
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


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception(  # pylint: disable=W0719
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    for i, _ in enumerate(sys.argv):
        abs_path: str = os.path.abspath(os.path.join(os.getcwd(), sys.argv[i]))
        warnings, init_dirs = get_init_dir_list_and_warnings(abs_path)
        fix_all_init_files(init_dirs)
