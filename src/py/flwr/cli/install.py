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

import typer
from typing_extensions import Annotated

from .config_utils import load_and_validate_with_defaults


def install(
    source: Annotated[
        Optional[Path],
        typer.Argument(metavar="source", help="The source FAB file to install."),
    ] = None,
    flwr_dir: Annotated[
        Optional[Path],
        typer.Option(help="The desired install path."),
    ] = None,
) -> None:
    """Install a Flower App Bundle.

    It can be ran with a single FAB file argument:

        ``flwr install ./target_project.fab``

    The target install directory can be specified with ``--flwr-dir``:

        ``flwr install ./target_project.fab --flwr-dir ./docs/flwr``

    This will install ``target_project`` to ``./docs/flwr/``. By default,
    ``flwr-dir`` is equal to:

        - ``$FLWR_HOME/`` if ``$FLWR_HOME`` is defined
        - ``$XDG_DATA_HOME/.flwr/`` if ``$XDG_DATA_HOME`` is defined
        - ``$HOME/.flwr/`` in all other cases
    """
    if source is None:
        source = Path(typer.prompt("Enter the source FAB file"))

    source = source.resolve()
    if not source.exists() or not source.is_file():
        typer.secho(
            f"❌ The source {source} does not exist or is not a file.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if source.suffix != ".fab":
        typer.secho(
            f"❌ The source {source} is not a `.fab` file.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    install_from_fab(source, flwr_dir)


def install_from_fab(fab_file: Path, flwr_dir: Optional[Path]) -> None:
    """Install from a FAB file after extracting and validating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(fab_file, "r") as zipf:
            zipf.extractall(tmpdir)
            tmpdir_path = Path(tmpdir)
            info_dir = tmpdir_path / ".info"
            if not info_dir.exists():
                typer.secho(
                    "❌ FAB file has incorrect format.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1)

            content_file = info_dir / "CONTENT"

            if not content_file.exists() or not _verify_hashes(
                content_file.read_text(), tmpdir_path
            ):
                typer.secho(
                    "❌ File hashes couldn't be verified.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1)

            shutil.rmtree(info_dir)

            validate_and_install(tmpdir_path, fab_file.stem, flwr_dir)


def validate_and_install(
    project_dir: Path, fab_name: str, flwr_dir: Optional[Path]
) -> None:
    """Validate TOML files and install the project to the desired directory."""
    config, _, _ = load_and_validate_with_defaults(project_dir / "pyproject.toml")

    if config is None:
        typer.secho(
            "❌ Invalid config inside FAB file.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    username = config["flower"]["publisher"]
    project_name = config["project"]["name"]
    version = config["project"]["version"]

    if fab_name != f"{username}.{project_name}.{version.replace('.', '-')}":
        typer.secho(
            "❌ FAB file has incorrect name, it should be "
            "`<username>.<project_name>.<version>.fab`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    install_dir: Path = (
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
        / project_name
        / version
    )
    if install_dir.exists() and not typer.confirm(
        typer.style(
            f"\n💬 {project_name} version {version} is already installed, "
            "do you want to reinstall it?",
            fg=typer.colors.MAGENTA,
            bold=True,
        )
    ):
        return

    install_dir.mkdir(parents=True, exist_ok=True)

    # Move contents from source directory
    for item in project_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, install_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, install_dir / item.name)

    typer.secho(
        f"🎊 Successfully installed {project_name} to {install_dir}.",
        fg=typer.colors.GREEN,
        bold=True,
    )


def _verify_hashes(list_content: str, tmpdir: Path) -> bool:
    """Verify file hashes based on the LIST content."""
    for line in list_content.strip().split("\n"):
        rel_path, hash_expected, _ = line.split(",")
        file_path = tmpdir / rel_path
        if not file_path.exists() or _get_sha256_hash(file_path) != hash_expected:
            return False
    return True


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
