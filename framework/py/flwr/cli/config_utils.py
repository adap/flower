# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility to validate the `pyproject.toml` file."""


from pathlib import Path
from typing import Any, Optional, Union

import tomli
import typer

from flwr.common.config import (
    fuse_dicts,
    get_fab_config,
    get_metadata_from_config,
    parse_config_args,
    validate_config,
)


def get_fab_metadata(fab_file: Union[Path, bytes]) -> tuple[str, str]:
    """Extract the fab_id and the fab_version from a FAB file or path.

    Parameters
    ----------
    fab_file : Union[Path, bytes]
        The Flower App Bundle file to validate and extract the metadata from.
        It can either be a path to the file or the file itself as bytes.

    Returns
    -------
    Tuple[str, str]
        The `fab_id` and `fab_version` of the given Flower App Bundle.
    """
    return get_metadata_from_config(get_fab_config(fab_file))


def load_and_validate(
    path: Optional[Path] = None,
    check_module: bool = True,
) -> tuple[Optional[dict[str, Any]], list[str], list[str]]:
    """Load and validate pyproject.toml as dict.

    Parameters
    ----------
    path : Optional[Path] (default: None)
        The path of the Flower App config file to load. By default it
        will try to use `pyproject.toml` inside the current directory.
    check_module: bool (default: True)
        Whether the validity of the Python module should be checked.
        This requires the project to be installed in the currently
        running environment. True by default.

    Returns
    -------
    Tuple[Optional[config], List[str], List[str]]
        A tuple with the optional config in case it exists and is valid
        and associated errors and warnings.
    """
    if path is None:
        path = Path.cwd() / "pyproject.toml"

    config = load(path)

    if config is None:
        errors = [
            "Project configuration could not be loaded. "
            "`pyproject.toml` does not exist."
        ]
        return (None, errors, [])

    is_valid, errors, warnings = validate_config(config, check_module, path.parent)

    if not is_valid:
        return (None, errors, warnings)

    return (config, errors, warnings)


def load(toml_path: Path) -> Optional[dict[str, Any]]:
    """Load pyproject.toml and return as dict."""
    if not toml_path.is_file():
        return None

    with toml_path.open("rb") as toml_file:
        try:
            return tomli.load(toml_file)
        except tomli.TOMLDecodeError:
            return None


def process_loaded_project_config(
    config: Union[dict[str, Any], None], errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    """Process and return the loaded project configuration.

    This function handles errors and warnings from the `load_and_validate` function,
    exits on critical issues, and returns the validated configuration.
    """
    if config is None:
        typer.secho(
            "Project configuration could not be loaded.\n"
            "pyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    typer.secho("Success", fg=typer.colors.GREEN)

    return config


def validate_federation_in_project_config(
    federation: Optional[str],
    config: dict[str, Any],
    overrides: Optional[list[str]] = None,
) -> tuple[str, dict[str, Any]]:
    """Validate the federation name in the Flower project configuration."""
    federation = federation or config["tool"]["flwr"]["federations"].get("default")

    if federation is None:
        typer.secho(
            "❌ No federation name was provided and the project's `pyproject.toml` "
            "doesn't declare a default federation (with an Exec API address or an "
            "`options.num-supernodes` value).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Validate the federation exists in the configuration
    federation_config = config["tool"]["flwr"]["federations"].get(federation)
    if federation_config is None:
        available_feds = {
            fed for fed in config["tool"]["flwr"]["federations"] if fed != "default"
        }
        typer.secho(
            f"❌ There is no `{federation}` federation declared in the "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Override the federation configuration if provided
    if overrides:
        overrides_dict = parse_config_args(overrides, flatten=False)
        federation_config = fuse_dicts(federation_config, overrides_dict)

    return federation, federation_config


def validate_certificate_in_federation_config(
    app: Path, federation_config: dict[str, Any]
) -> tuple[bool, Optional[bytes]]:
    """Validate the certificates in the Flower project configuration."""
    insecure_str = federation_config.get("insecure")
    if root_certificates := federation_config.get("root-certificates"):
        root_certificates_bytes = (app / root_certificates).read_bytes()
        if insecure := bool(insecure_str):
            typer.secho(
                "❌ `root-certificates` were provided but the `insecure` parameter "
                "is set to `True`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
    else:
        root_certificates_bytes = None
        if insecure_str is None:
            typer.secho(
                "❌ To disable TLS, set `insecure = true` in `pyproject.toml`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
        if not (insecure := bool(insecure_str)):
            typer.secho(
                "❌ No certificate were given yet `insecure` is set to `False`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

    return insecure, root_certificates_bytes


def exit_if_no_address(federation_config: dict[str, Any], cmd: str) -> None:
    """Exit if the provided federation_config has no "address" key."""
    if "address" not in federation_config:
        typer.secho(
            f"❌ `flwr {cmd}` currently works with a SuperLink. Ensure that the correct"
            "SuperLink (Exec API) address is provided in `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)
