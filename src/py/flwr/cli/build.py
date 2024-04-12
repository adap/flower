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
"""Flower command line interface `example` command."""

import hashlib
import os
import pathspec
import typer
import zipfile
from pathlib import Path


def build(
    directory: Path = typer.Option(Path.cwd(), help="The directory to zip")
) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        typer.echo(f"The path {directory} is not a valid directory.")
        raise typer.Exit(code=1)

    # Load .gitignore rules if present
    ignore_spec = _load_gitignore(directory)

    # Set the name of the zip file
    zip_filename = f"{directory.name}.fab"
    list_file_content = ""

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory, topdown=True):
            # Filter directories and files based on .gitignore
            dirs[:] = [d for d in dirs if not ignore_spec.match_file(Path(root) / d)]
            files = [
                f
                for f in files
                if not ignore_spec.match_file(Path(root) / f) and f != zip_filename
            ]

            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(directory)
                zipf.write(file_path, arc_path)

                # Calculate file info
                sha256_hash = _get_sha256_hash(file_path)
                file_size_bits = os.path.getsize(file_path) * 8  # size in bits
                list_file_content += f"{arc_path},{sha256_hash},{file_size_bits}\n"

        # Add .info/LIST to the zip file
        zipf.writestr(".info/LIST", list_file_content)

    typer.secho(f"Bundled FAB into {zip_filename}", fg=typer.colors.GREEN)


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
        with open(gitignore_path, "r") as file:
            patterns.extend(file.readlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
