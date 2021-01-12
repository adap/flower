#!/usr/bin/env python3
# Copyright 2020 Adap GmbH. All Rights Reserved.

"""This File will check the specified directory for any missing __init__.py
files."""

import os
import re
import sys


def check_missing_init_files(absolute_path: str) -> None:
    """Searches through the specified absolute_path and looks for missing __init__.py
    files."""
    path = os.walk(absolute_path)
    warning_list = []
    ignore_list = ["__pycache__$", ".pytest_cache.*$", "dist", "flwr.egg-info$"]
    should_skip: bool = False

    for dir_path, _, files_in_dir in path:
        # As some directories are automatically
        # generated we are going to ignore them
        for ignore_word in ignore_list:
            if re.search(ignore_word, dir_path) is not None:
                should_skip = True
                break
        if should_skip:
            should_skip = False
            continue

        # If no init is found in current directory add a warning_message to warning_list
        if not any(filename == "__init__.py" for filename in files_in_dir):
            warning_message = "- " + dir_path
            warning_list.append(warning_message)

    if len(warning_list) > 0:
        print("Could not find '__init__.py' in the   following directories:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Please provide path to directory as in `init_py_check.py src/py/adap`"
        )
    abs_path: str = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    check_missing_init_files(abs_path)
