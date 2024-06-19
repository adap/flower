# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
"""Fix copyright notices in all Python files of a given directory.

Example:
    python -m flwr_tool.fix_copyright src/py/flwr
"""


import os
import sys
from pathlib import Path
from typing import List

from flwr_tool.check_copyright import COPYRIGHT_FORMAT, _get_file_creation_year
from flwr_tool.init_py_check import get_init_dir_list_and_warnings


def _insert_or_edit_copyright(py_file: Path) -> None:
    contents = py_file.read_text()
    lines = contents.splitlines()
    creation_year = _get_file_creation_year(str(py_file.absolute()))
    expected_copyright = COPYRIGHT_FORMAT.format(creation_year)

    if expected_copyright not in contents:
        if "Copyright" in lines[0]:
            end_index = 0
            for idx, line in enumerate(lines):
                if (
                    line.strip()
                    == COPYRIGHT_FORMAT.rsplit("\n", maxsplit=1)[-1].strip()
                ):
                    end_index = idx + 1
                    break
            lines = lines[end_index:]

        lines.insert(0, expected_copyright)
    py_file.write_text("\n".join(lines), newline="\n")


def _fix_copyright(dir_list: List[str]) -> None:
    for valid_dir in dir_list:
        dir_path = Path(valid_dir)
        for py_file in dir_path.glob("*.py"):
            _insert_or_edit_copyright(py_file)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception(  # pylint: disable=W0719
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    for i, _ in enumerate(sys.argv):
        abs_path: str = os.path.abspath(os.path.join(os.getcwd(), sys.argv[i]))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        _fix_copyright(init_dirs)
