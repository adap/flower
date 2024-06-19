# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
"""Check if copyright notices are present in all Python files.

Example:
    python -m flwr_tool.check_copyright src/py/flwr
"""


import os
from pathlib import Path
import sys
from typing import List

from flwr_tool.init_py_check import get_init_dir_list_and_warnings
from flwr_tool.check_copyright import _get_file_creation_year, COPYRIGHT_FORMAT


def _check_copyright(dir_list: List[str]) -> None:
    warning_list = []
    for valid_dir in dir_list:
        dir_path = Path(valid_dir)
        for py_file in dir_path.glob("*.py"):
            contents = py_file.read_text()
            lines = contents.splitlines()
            creation_year = _get_file_creation_year(str(py_file.absolute()))
            expected_copyright = COPYRIGHT_FORMAT.format(creation_year)

            if expected_copyright not in contents:
                if "Copyright" in lines[0]:
                    end_index = 0
                    for i, line in enumerate(lines):
                        if line.strip() == COPYRIGHT_FORMAT.split("\n")[-1].strip():
                            end_index = i + 1
                            break
                    lines = lines[end_index:]

                lines.insert(0, expected_copyright)
            py_file.write_text("\n".join(lines))

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
