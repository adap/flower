# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
"""Check provided directory and sub-directories for missing __init__.py files.

Example:
    python -m flwr_tool.init_py_check src/py/flwr
"""


import os
import re
import sys


def check_missing_init_files(absolute_path: str) -> None:
    """Search absolute_path and look for missing __init__.py files."""
    path = os.walk(absolute_path)
    warning_list = []
    ignore_list = ["__pycache__$", ".pytest_cache.*$", "dist", "flwr.egg-info$"]

    for dir_path, _, files_in_dir in path:
        # As some directories are automatically generated we are going to ignore them
        if any(re.search(iw, dir_path) is not None for iw in ignore_list):
            continue

        # If no init is found in current directory add a warning_message to warning_list
        if not any(filename == "__init__.py" for filename in files_in_dir):
            warning_message = "- " + dir_path
            warning_list.append(warning_message)

    if len(warning_list) > 0:
        print("Could not find '__init__.py' in the following directories:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception(  # pylint: disable=W0719
            "Please provide at least one directory path relative to your current working directory."
        )
    for i, _ in enumerate(sys.argv):
        abs_path: str = os.path.abspath(os.path.join(os.getcwd(), sys.argv[i]))
        check_missing_init_files(abs_path)
