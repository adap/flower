# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
"""Check provided directory and sub-directories for missing __init__.py files.

Example:
    python -m flwr_tool.init_py_check src/py/flwr
"""


import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def get_init_dir_list_and_warnings(absolute_path: str) -> Tuple[List[str], List[str]]:
    """Search given path and return list of dirs containing __init__.py files."""
    path = os.walk(absolute_path)
    warning_list = []
    dir_list = []
    ignore_list = ["__pycache__$", ".pytest_cache.*$", "dist", "flwr.egg-info$"]

    for dir_path, _, files_in_dir in path:
        # As some directories are automatically generated we are going to ignore them
        if any(re.search(iw, dir_path) is not None for iw in ignore_list):
            continue

        # If no init is found in current directory add a warning_message to warning_list
        if not any(filename == "__init__.py" for filename in files_in_dir):
            warning_message = "- " + dir_path
            warning_list.append(warning_message)
        else:
            dir_list.append(dir_path)
    return warning_list, dir_list


def check_missing_init_files(absolute_path: str) -> List[str]:
    """Search absolute_path and look for missing __init__.py files."""
    warning_list, dir_list = get_init_dir_list_and_warnings(absolute_path)

    if len(warning_list) > 0:
        print("Could not find '__init__.py' in the following directories:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)

    return dir_list


def get_all_var_list(init_dir: str) -> Tuple[Path, List[str], List[str]]:
    """Get the __all__ list of a __init__.py file.

    The function returns the path of the '__init__.py' file of the given dir, as well as
    the list itself, and the list of lines corresponding to the list.
    """
    init_file = Path(init_dir) / "__init__.py"
    all_lines = []
    all_list = []
    capture = False
    for line in init_file.read_text().splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("__all__"):
            capture = True
        if capture:
            all_lines.append(line)
            if stripped_line.endswith("]"):
                capture = False
                break

    if all_lines:
        all_string = "".join(all_lines)
        all_list = ast.literal_eval(all_string.split("=", 1)[1].strip())

    return init_file, all_list, all_lines


def check_all_init_files(dir_list: List[str]) -> None:
    """Check if __all__ is in alphabetical order in __init__.py files."""
    warning_list = []

    for init_dir in dir_list:
        init_file, all_list, _ = get_all_var_list(init_dir)

        if all_list and not all_list == sorted(all_list):
            warning_message = "- " + str(init_file)
            warning_list.append(warning_message)

    if len(warning_list) > 0:
        print(
            "'__all__' lists in the following '__init__.py' files are "
            "incorrectly sorted:"
        )
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
        init_dirs = check_missing_init_files(abs_path)
        check_all_init_files(init_dirs)
