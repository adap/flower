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
"""Flower command line interface utils."""

import hashlib
import re
from pathlib import Path
from typing import Callable, List, Optional, cast

import typer


def prompt_text(
    text: str,
    predicate: Callable[[str], bool] = lambda _: True,
    default: Optional[str] = None,
) -> str:
    """Ask user to enter text input."""
    while True:
        result = typer.prompt(
            typer.style(f"\n💬 {text}", fg=typer.colors.MAGENTA, bold=True),
            default=default,
        )
        if predicate(result) and len(result) > 0:
            break
        print(typer.style("❌ Invalid entry", fg=typer.colors.RED, bold=True))

    return cast(str, result)


def prompt_options(text: str, options: List[str]) -> str:
    """Ask user to select one of the given options and return the selected item."""
    # Turn options into a list with index as in " [ 0] quickstart-pytorch"
    options_formatted = [
        " [ "
        + typer.style(index, fg=typer.colors.GREEN, bold=True)
        + "]"
        + f" {typer.style(name, fg=typer.colors.WHITE, bold=True)}"
        for index, name in enumerate(options)
    ]

    while True:
        index = typer.prompt(
            "\n"
            + typer.style(f"💬 {text}", fg=typer.colors.MAGENTA, bold=True)
            + "\n\n"
            + "\n".join(options_formatted)
            + "\n\n\n"
        )
        try:
            options[int(index)]  # pylint: disable=expression-not-assigned
            break
        except IndexError:
            print(typer.style("❌ Index out of range", fg=typer.colors.RED, bold=True))
            continue
        except ValueError:
            print(
                typer.style("❌ Please choose a number", fg=typer.colors.RED, bold=True)
            )
            continue

    result = options[int(index)]
    return result


def is_valid_project_name(name: str) -> bool:
    """Check if the given string is a valid Python project name.

    A valid project name must start with a letter and can only contain letters, digits,
    and hyphens.
    """
    if not name:
        return False

    # Check if the first character is a letter
    if not name[0].isalpha():
        return False

    # Check if the rest of the characters are valid (letter, digit, or dash)
    for char in name[1:]:
        if not (char.isalnum() or char in "-"):
            return False

    return True


def sanitize_project_name(name: str) -> str:
    """Sanitize the given string to make it a valid Python project name.

    This version replaces spaces, dots, slashes, and underscores with dashes, removes
    any characters not allowed in Python project names, makes the string lowercase, and
    ensures it starts with a valid character.
    """
    # Replace whitespace with '_'
    name_with_hyphens = re.sub(r"[ ./_]", "-", name)

    # Allowed characters in a module name: letters, digits, underscore
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )

    # Make the string lowercase
    sanitized_name = name_with_hyphens.lower()

    # Remove any characters not allowed in Python module names
    sanitized_name = "".join(c for c in sanitized_name if c in allowed_chars)

    # Ensure the first character is a letter or underscore
    while sanitized_name and (
        sanitized_name[0].isdigit() or sanitized_name[0] not in allowed_chars
    ):
        sanitized_name = sanitized_name[1:]

    return sanitized_name


def get_sha256_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64kB blocks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()
