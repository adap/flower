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
"""Utility to validate the `pyproject.toml` file."""

import zipfile
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Optional, Union, get_args

import tomli

from flwr.common import object_ref
from flwr.common.typing import UserConfigValue


def get_fab_config(fab_file: Union[Path, bytes]) -> dict[str, Any]:
    """Extract the config from a FAB file or path.

    Parameters
    ----------
    fab_file : Union[Path, bytes]
        The Flower App Bundle file to validate and extract the metadata from.
        It can either be a path to the file or the file itself as bytes.

    Returns
    -------
    Dict[str, Any]
        The `config` of the given Flower App Bundle.
    """
    fab_file_archive: Union[Path, IO[bytes]]
    if isinstance(fab_file, bytes):
        fab_file_archive = BytesIO(fab_file)
    elif isinstance(fab_file, Path):
        fab_file_archive = fab_file
    else:
        raise ValueError("fab_file must be either a Path or bytes")

    with zipfile.ZipFile(fab_file_archive, "r") as zipf:
        with zipf.open("pyproject.toml") as file:
            toml_content = file.read().decode("utf-8")

        conf = load_from_string(toml_content)
        if conf is None:
            raise ValueError("Invalid TOML content in pyproject.toml")

        is_valid, errors, _ = validate(conf, check_module=False)
        if not is_valid:
            raise ValueError(errors)

        return conf


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
    conf = get_fab_config(fab_file)

    return (
        f"{conf['tool']['flwr']['app']['publisher']}/{conf['project']['name']}",
        conf["project"]["version"],
    )


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

    is_valid, errors, warnings = validate(config, check_module, path.parent)

    if not is_valid:
        return (None, errors, warnings)

    return (config, errors, warnings)


def load(toml_path: Path) -> Optional[dict[str, Any]]:
    """Load pyproject.toml and return as dict."""
    if not toml_path.is_file():
        return None

    with toml_path.open(encoding="utf-8") as toml_file:
        return load_from_string(toml_file.read())


def _validate_run_config(config_dict: dict[str, Any], errors: list[str]) -> None:
    for key, value in config_dict.items():
        if isinstance(value, dict):
            _validate_run_config(config_dict[key], errors)
        elif not isinstance(value, get_args(UserConfigValue)):
            raise ValueError(
                f"The value for key {key} needs to be of type `int`, `float`, "
                "`bool, `str`, or  a `dict` of those.",
            )


# pylint: disable=too-many-branches
def validate_fields(config: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Validate pyproject.toml fields."""
    errors = []
    warnings = []

    if "project" not in config:
        errors.append("Missing [project] section")
    else:
        if "name" not in config["project"]:
            errors.append('Property "name" missing in [project]')
        if "version" not in config["project"]:
            errors.append('Property "version" missing in [project]')
        if "description" not in config["project"]:
            warnings.append('Recommended property "description" missing in [project]')
        if "license" not in config["project"]:
            warnings.append('Recommended property "license" missing in [project]')
        if "authors" not in config["project"]:
            warnings.append('Recommended property "authors" missing in [project]')

    if (
        "tool" not in config
        or "flwr" not in config["tool"]
        or "app" not in config["tool"]["flwr"]
    ):
        errors.append("Missing [tool.flwr.app] section")
    else:
        if "publisher" not in config["tool"]["flwr"]["app"]:
            errors.append('Property "publisher" missing in [tool.flwr.app]')
        if "config" in config["tool"]["flwr"]["app"]:
            _validate_run_config(config["tool"]["flwr"]["app"]["config"], errors)
        if "components" not in config["tool"]["flwr"]["app"]:
            errors.append("Missing [tool.flwr.app.components] section")
        else:
            if "serverapp" not in config["tool"]["flwr"]["app"]["components"]:
                errors.append(
                    'Property "serverapp" missing in [tool.flwr.app.components]'
                )
            if "clientapp" not in config["tool"]["flwr"]["app"]["components"]:
                errors.append(
                    'Property "clientapp" missing in [tool.flwr.app.components]'
                )

    return len(errors) == 0, errors, warnings


def validate(
    config: dict[str, Any],
    check_module: bool = True,
    project_dir: Optional[Union[str, Path]] = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate pyproject.toml."""
    is_valid, errors, warnings = validate_fields(config)

    if not is_valid:
        return False, errors, warnings

    # Validate serverapp
    serverapp_ref = config["tool"]["flwr"]["app"]["components"]["serverapp"]
    is_valid, reason = object_ref.validate(serverapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], []

    # Validate clientapp
    clientapp_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]
    is_valid, reason = object_ref.validate(clientapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], []

    return True, [], []


def load_from_string(toml_content: str) -> Optional[dict[str, Any]]:
    """Load TOML content from a string and return as dict."""
    try:
        data = tomli.loads(toml_content)
        return data
    except tomli.TOMLDecodeError:
        return None
