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
"""Flower command line interface `build` command."""

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional

import jwt
import pathspec
import typer
from typing_extensions import Annotated


def build(
    directory: Annotated[
        Optional[Path], typer.Option(help="The to bundle into a FAB")
    ] = None,
    signed: Annotated[bool, typer.Option(help="Flag to sign the FAB")] = False,
) -> None:
    """Build a Flower project."""
    if directory is None:
        directory = Path.cwd()

    directory = directory.resolve()
    if not directory.is_dir():
        typer.echo(f"The path {directory} is not a valid directory.")
        raise typer.Exit(code=1)

    # Load .gitignore rules if present
    ignore_spec = _load_gitignore(directory)

    # Set the name of the zip file
    fab_filename = f"{directory.name}.fab"
    list_file_content = ""

    with zipfile.ZipFile(fab_filename, "w", zipfile.ZIP_DEFLATED) as fab_file:
        for root, _, files in os.walk(directory, topdown=True):
            # Filter directories and files based on .gitignore
            files = [
                f
                for f in files
                if not ignore_spec.match_file(Path(root) / f) and f != fab_filename
            ]

            for file in files:
                file_path = Path(root) / file
                archive_path = file_path.relative_to(directory)
                fab_file.write(file_path, archive_path)

                # Calculate file info
                sha256_hash = _get_sha256_hash(file_path)
                file_size_bits = os.path.getsize(file_path) * 8  # size in bits
                list_file_content += f"{archive_path},{sha256_hash},{file_size_bits}\n"

        # Add LIST and LIST.jwt to the zip file
        fab_file.writestr(".info/LIST", list_file_content)

        # Optionally sign the LIST file
        if signed:
            secret_key = typer.prompt(
                "Enter the secret key to sign the LIST file", hide_input=True
            )
            signed_token = _sign_content(list_file_content, secret_key)
            fab_file.writestr(".info/LIST.jwt", signed_token)

    typer.secho(f"Bundled FAB into {fab_filename}", fg=typer.colors.GREEN)


def _get_sha256_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64kB blocks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def _load_gitignore(directory: Path) -> pathspec.PathSpec:
    """Load and parse .gitignore file, returning a pathspec."""
    gitignore_path = directory / ".gitignore"
    patterns = ["__pycache__/"]  # Default pattern
    if gitignore_path.exists():
        with open(gitignore_path) as file:
            patterns.extend(file.readlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _sign_content(content: str, secret_key: str) -> str:
    """Signs the content using JWT and returns the token."""
    payload = {"data": content}
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token
