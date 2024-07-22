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
"""Provide functions for managing global Flower config."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, get_args

import tomli

from flwr.cli.config_utils import validate_fields
from flwr.common.constant import APP_DIR, FAB_CONFIG_FILE, FLWR_HOME
from flwr.common.typing import Run, UserConfig, UserConfigValue


def get_flwr_dir(provided_path: Optional[str] = None) -> Path:
    """Return the Flower home directory based on env variables."""
    if provided_path is None or not Path(provided_path).is_dir():
        return Path(
            os.getenv(
                FLWR_HOME,
                Path(f"{os.getenv('XDG_DATA_HOME', os.getenv('HOME'))}") / ".flwr",
            )
        )
    return Path(provided_path).absolute()


def get_project_dir(
    fab_id: str, fab_version: str, flwr_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Return the project directory based on the given fab_id and fab_version."""
    # Check the fab_id
    if fab_id.count("/") != 1:
        raise ValueError(
            f"Invalid FAB ID: {fab_id}",
        )
    publisher, project_name = fab_id.split("/")
    if flwr_dir is None:
        flwr_dir = get_flwr_dir()
    return Path(flwr_dir) / APP_DIR / publisher / project_name / fab_version


def get_project_config(project_dir: Union[str, Path]) -> Dict[str, Any]:
    """Return pyproject.toml in the given project directory."""
    # Load pyproject.toml file
    toml_path = Path(project_dir) / FAB_CONFIG_FILE
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {FAB_CONFIG_FILE} in {project_dir}",
        )
    with toml_path.open(encoding="utf-8") as toml_file:
        config = tomli.loads(toml_file.read())

    # Validate pyproject.toml fields
    is_valid, errors, _ = validate_fields(config)
    if not is_valid:
        error_msg = "\n".join([f"  - {error}" for error in errors])
        raise ValueError(
            f"Invalid {FAB_CONFIG_FILE}:\n{error_msg}",
        )

    return config


def _fuse_dicts(
    main_dict: UserConfig,
    override_dict: UserConfig,
) -> UserConfig:
    fused_dict = main_dict.copy()

    for key, value in override_dict.items():
        if key in main_dict:
            fused_dict[key] = value

    return fused_dict


def get_fused_config_from_dir(
    project_dir: Path, override_config: UserConfig
) -> UserConfig:
    """Merge the overrides from a given dict with the config from a Flower App."""
    default_config = get_project_config(project_dir)["tool"]["flwr"]["app"].get(
        "config", {}
    )
    flat_default_config = flatten_dict(default_config)

    return _fuse_dicts(flat_default_config, override_config)


def get_fused_config(run: Run, flwr_dir: Optional[Path]) -> UserConfig:
    """Merge the overrides from a `Run` with the config from a FAB.

    Get the config using the fab_id and the fab_version, remove the nesting by adding
    the nested keys as prefixes separated by dots, and fuse it with the override dict.
    """
    if not run.fab_id or not run.fab_version:
        return {}

    project_dir = get_project_dir(run.fab_id, run.fab_version, flwr_dir)

    return get_fused_config_from_dir(project_dir, run.override_config)


def flatten_dict(raw_dict: Dict[str, Any], parent_key: str = "") -> UserConfig:
    """Flatten dict by joining nested keys with a given separator."""
    if raw_dict is None:
        return {}

    items: List[Tuple[str, UserConfigValue]] = []
    separator: str = "."
    for k, v in raw_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        elif isinstance(v, get_args(UserConfigValue)):
            items.append((new_key, cast(UserConfigValue, v)))
        else:
            raise ValueError(
                f"The value for key {k} needs to be of type `int`, `float`, "
                "`bool, `str`, or  a `dict` of those.",
            )
    return dict(items)


def parse_config_args(
    config: Optional[List[str]],
    separator: str = ",",
) -> UserConfig:
    """Parse separator separated list of key-value pairs separated by '='."""
    overrides: UserConfig = {}

    if config is None:
        return overrides

    for config_line in config:
        if config_line:
            overrides_list = config_line.split(separator)
            if (
                len(overrides_list) == 1
                and "=" not in overrides_list
                and overrides_list[0].endswith(".toml")
            ):
                with Path(overrides_list[0]).open("rb") as config_file:
                    overrides = flatten_dict(tomli.load(config_file))
            else:
                toml_str = "\n".join(overrides_list)
                overrides.update(tomli.loads(toml_str))

    return overrides
