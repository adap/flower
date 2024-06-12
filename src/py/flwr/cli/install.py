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


import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import IO, Optional, Union

import typer
from typing_extensions import Annotated

from .config_utils import get_flower_home, load_and_validate
from .utils import get_sha256_hash


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


def install_from_fab(
    fab_file: Union[Path, bytes], flwr_dir: Optional[Path], skip_prompt: bool = False
) -> Path:
    """Install from a FAB file after extracting and validating."""
    fab_file_archive: Union[Path, IO[bytes]]
    fab_name: Optional[str]
    if isinstance(fab_file, bytes):
        fab_file_archive = BytesIO(fab_file)
        fab_name = None
    elif isinstance(fab_file, Path):
        fab_file_archive = fab_file
        fab_name = fab_file.stem
    else:
        raise ValueError("fab_file must be either a Path or bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(fab_file_archive, "r") as zipf:
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

            installed_path = validate_and_install(
                tmpdir_path, fab_name, flwr_dir, skip_prompt
            )

    return installed_path


def validate_and_install(
    project_dir: Path,
    fab_name: Optional[str],
    flwr_dir: Optional[Path],
    skip_prompt: bool = False,
) -> Path:
    """Validate TOML files and install the project to the desired directory."""
    config, _, _ = load_and_validate(project_dir / "pyproject.toml", check_module=False)

    if config is None:
        typer.secho(
            "❌ Invalid config inside FAB file.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    publisher = config["flower"]["publisher"]
    project_name = config["project"]["name"]
    version = config["project"]["version"]

    if (
        fab_name
        and fab_name != f"{publisher}.{project_name}.{version.replace('.', '-')}"
    ):
        typer.secho(
            "❌ FAB file has incorrect name. The file name must follow the format "
            "`<publisher>.<project_name>.<version>.fab`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    install_dir: Path = (
        (get_flower_home() if not flwr_dir else flwr_dir)
        / "apps"
        / publisher
        / project_name
        / version
    )
    if install_dir.exists() and not skip_prompt:
        if not typer.confirm(
            typer.style(
                f"\n💬 {project_name} version {version} is already installed, "
                "do you want to reinstall it?",
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        ):
            return install_dir

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

    return install_dir


def _verify_hashes(list_content: str, tmpdir: Path) -> bool:
    """Verify file hashes based on the LIST content."""
    for line in list_content.strip().split("\n"):
        rel_path, hash_expected, _ = line.split(",")
        file_path = tmpdir / rel_path
        if not file_path.exists() or get_sha256_hash(file_path) != hash_expected:
            return False
    return True
