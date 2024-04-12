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
"""Flower command line interface `install` command."""


import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import jwt
import tomli
import typer
from typing_extensions import Annotated

from .flower_toml import load, validate


def install(
    source: Annotated[
        Optional[Path],
        typer.Argument(
            metavar="source", help="The source FAB file or directory to install"
        ),
    ] = None,
    flwr_dir: Annotated[
        Optional[Path],
        typer.Option(help="The desired install path"),
    ] = None,
) -> None:
    """Install a Flower project from a FAB file or a directory."""
    if source is None:
        source = Path(typer.prompt("Enter the source FAB file or directory path"))

    source = source.resolve()
    if not source.exists():
        typer.echo(f"The source {source} does not exist.")
        raise typer.Exit(code=1)

    if source.is_dir():
        install_from_directory(source, flwr_dir)
    else:
        install_from_fab(source, flwr_dir)


def install_from_directory(directory: Path, flwr_dir: Optional[Path]) -> None:
    """Install directly from a directory."""
    validate_and_install(directory, flwr_dir)


def install_from_fab(fab_file: Path, flwr_dir: Optional[Path]) -> None:
    """Install from a FAB file after extracting and validating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(fab_file, "r") as zipf:
            zipf.extractall(tmpdir)
            tmpdir_path = Path(tmpdir)
            list_path = tmpdir_path / ".info/LIST"
            jwt_path = tmpdir_path / ".info/LIST.jwt"

            if jwt_path.exists():
                secret_key = typer.prompt(
                    "Enter the public key to verify the LIST file", hide_input=True
                )
                if not _verify_jwt_signature(
                    list_path.read_text(), jwt_path.read_text(), secret_key
                ):
                    typer.echo("Failed to verify the LIST file signature.")
                    raise typer.Exit(code=1)

            if not _verify_hashes(list_path.read_text(), tmpdir_path):
                typer.echo("File hashes do not match.")
                raise typer.Exit(code=1)

            validate_and_install(tmpdir_path, flwr_dir)


def validate_and_install(project_dir: Path, flwr_dir: Optional[Path]) -> None:
    """Validate TOML files and install the project to the desired directory."""
    config = load(str(project_dir / "flower.toml"))
    if config is None:
        typer.secho(
            "Project configuration could not be loaded. flower.toml does not exist."
        )
        raise typer.Exit(code=1)

    if not validate(config):
        typer.secho("Project configuration is invalid.")
        raise typer.Exit(code=1)

    if "provider" not in config["flower"]:
        typer.secho("Project configuration is missing required 'provider' field.")
        raise typer.Exit(code=1)

    username = config["flower"]["provider"]
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.exists():
        typer.secho("`pyproject.toml` not found.")
        raise typer.Exit(code=1)

    with pyproject_path.open(encoding="utf-8") as toml_file:
        data = tomli.loads(toml_file.read())
        if "project" not in data or "version" not in data["project"]:
            typer.secho("Couldn't find `version` in `pyproject.toml`.")
            raise typer.Exit(code=1)

        version = data["project"]["version"]

    install_dir = (
        (
            Path(
                os.getenv(
                    "FLWR_HOME",
                    f"{os.getenv('XDG_DATA_HOME', os.getenv('HOME'))}/.flwr",
                )
            )
            if not flwr_dir
            else flwr_dir
        )
        / "apps"
        / username
        / project_dir.stem
        / version
    )
    install_dir.mkdir(parents=True, exist_ok=True)

    # Move contents from source directory
    for item in project_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, install_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, install_dir / item.name)

    typer.echo(f"Successfully installed {project_dir.stem} to {install_dir}")


def _verify_hashes(list_content: str, tmpdir: Path) -> bool:
    """Verify file hashes based on the LIST content."""
    for line in list_content.strip().split("\n"):
        rel_path, hash_expected, _ = line.split(",")
        file_path = tmpdir / rel_path
        if not file_path.exists() or _get_sha256_hash(file_path) != hash_expected:
            return False
    return True


def _verify_jwt_signature(list_content: str, jwt_token: str, public_key: str) -> bool:
    """Verify the JWT signature."""
    try:
        decoded = jwt.decode(jwt_token, public_key, algorithms=["HS256"])
        return bool(decoded["data"] == list_content)
    except jwt.exceptions.InvalidTokenError:
        return False


def _get_sha256_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()
