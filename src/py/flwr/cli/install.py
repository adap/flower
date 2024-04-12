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
import jwt
import os
import shutil
import tempfile
import typer
import zipfile
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import tomli

from .flower_toml import load, validate


def install(
    fab_file: Annotated[
        Optional[Path],
        typer.Argument(metavar="project_name", help="The name of the project"),
    ] = None,
    flwr_dir: Annotated[
        Optional[Path],
        typer.Option(help="The desired install path"),
    ] = None,
) -> None:
    """Install a Flower project from a FAB file."""
    if fab_file is None:
        fab_file = Path(typer.prompt("FAB file to install"))

    fab_file = fab_file.resolve()
    if not fab_file.is_file():
        typer.echo(f"The file {fab_file} does not exist.")
        raise typer.Exit(code=1)

    # Create a temporary directory to extract the FAB file
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(fab_file, "r") as zipf:
            zipf.extractall(tmpdir)
            list_path = Path(tmpdir) / ".info/LIST"
            jwt_path = Path(tmpdir) / ".info/LIST.jwt"

            # Read the LIST content
            with open(list_path, "r") as file:
                list_content = file.read()

            # Verify the LIST file
            if jwt_path.exists():
                secret_key = typer.prompt(
                    "Enter the public key to verify the LIST file", hide_input=True
                )
                if not _verify_jwt_signature(
                    list_content, jwt_path.read_text(), secret_key
                ):
                    typer.echo("Failed to verify the LIST file signature.")
                    raise typer.Exit(code=1)
            else:
                if not _verify_hashes(list_content, Path(tmpdir)):
                    typer.echo("File hashes do not match.")
                    raise typer.Exit(code=1)

        config = load(str(Path(tmpdir) / "flower.toml"))
        if config is None:
            typer.secho(
                "Project configuration could not be loaded. flower.toml does not exist."
            )
            raise typer.Exit(code=1)

        is_valid, _, _ = validate(config)

        if not is_valid:
            typer.secho("Project configuration is invalid.")
            raise typer.Exit(code=1)

        if "provider" not in config["flower"]:
            typer.secho("Project configuration is invalid.")
            raise typer.Exit(code=1)

        username = config["flower"]["provider"]

        if not (Path(tmpdir) / "pyproject.toml").exists():
            typer.secho("`pyproject.toml` not found.")
            raise typer.Exit(code=1)

        with (Path(tmpdir) / "pyproject.toml").open(encoding="utf-8") as toml_file:
            data = tomli.loads(toml_file.read())
            if "project" not in data or "version" not in data["project"]:
                typer.secho("Couldn't find `version` in `pyproject.toml`.")
                raise typer.Exit(code=1)

            version = data["project"]["version"]

        # Determine installation directory
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
            / fab_file.stem
            / version
        )
        if not install_dir.exists():
            install_dir.mkdir(parents=True, exist_ok=True)

        # Move contents from temporary directory
        for item in Path(tmpdir).iterdir():
            shutil.move(str(item), str(install_dir))

    typer.echo(f"Successfully installed {fab_file.stem} to {install_dir}")


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
        return decoded["data"] == list_content
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
