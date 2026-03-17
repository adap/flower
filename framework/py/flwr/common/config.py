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
"""Provide functions for managing global Flower config."""


import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Any, TypeVar, cast, get_args

import click
import tomli
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from flwr.app.user_config import UserConfig, UserConfigValue
from flwr.common.constant import APP_DIR, FAB_CONFIG_FILE, FAB_HASH_TRUNCATION
from flwr.common.typing import Run
from flwr.supercore.utils import get_flwr_home

from . import ConfigRecord, object_ref

T_dict = TypeVar("T_dict", bound=dict[str, Any])  # pylint: disable=invalid-name


@dataclass(frozen=True)
class FabFormatMetadata:
    """Normalized metadata derived during FAB build and returned to callers."""

    fab_format_version: int
    flwr_version_min: str | None
    flwr_version_target: str | None
    flwr_version_max: str | None


class _FabFormatRules:
    """Base ruleset for a specific fab_format_version."""

    def normalize_and_validate_metadata(
        self, config: dict[str, Any]
    ) -> FabFormatMetadata:
        """Validate and derive metadata for this fab_format_version."""
        raise NotImplementedError

    def validate_fab_contents(
        self, config: dict[str, Any], filtered_paths: list[str]
    ) -> None:
        """Validate the final set of files that will be written into the FAB."""
        del config, filtered_paths


def _parse_version(value: str, field_name: str) -> Version:
    """Parse a version string and raise a consistent config error if invalid."""
    try:
        return Version(value)
    except InvalidVersion as err:
        raise ValueError(f"Invalid {field_name}: expected a valid version.") from err


def _get_flwr_requirement(config: dict[str, Any]) -> Requirement | None:
    """Return the unique `flwr` dependency declared in `[project].dependencies`."""
    dependencies = config.get("project", {}).get("dependencies")
    if dependencies is None:
        return None

    if not isinstance(dependencies, list):
        raise ValueError(
            'Invalid [project].dependencies: expected a list containing a "flwr" '
            "dependency declaration."
        )

    flwr_requirements: list[Requirement] = []
    for dependency in dependencies:
        if not isinstance(dependency, str):
            raise ValueError(
                "Invalid [project].dependencies: dependency entries must be strings."
            )
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as err:
            raise ValueError(
                f'Invalid dependency declaration "{dependency}": {err}'
            ) from err

        if requirement.name == "flwr":
            flwr_requirements.append(requirement)

    if not flwr_requirements:
        return None

    if len(flwr_requirements) > 1:
        raise ValueError('Multiple "flwr" dependency declarations are not supported.')

    return flwr_requirements[0]


def _derive_flwr_version_bounds(
    requirement: Requirement,
) -> tuple[Version, Version | None]:
    """Derive supported Flower version bounds from the `flwr` dependency."""
    lower: Version | None = None
    upper: Version | None = None

    for specifier in requirement.specifier:
        version = _parse_version(specifier.version, '"flwr" dependency specifier')
        if specifier.operator == ">=":
            if lower is not None and lower != version:
                raise ValueError(
                    'Unsupported "flwr" dependency specifier: multiple lower bounds '
                    "are not supported."
                )
            lower = version
        elif specifier.operator == "<=":
            if upper is not None and upper != version:
                raise ValueError(
                    'Unsupported "flwr" dependency specifier: multiple upper bounds '
                    "are not supported."
                )
            upper = version
        else:
            raise ValueError(
                'Unsupported "flwr" dependency specifier '
                f'"{specifier}" in requirement "{requirement}". '
                "For fab_format_version >= 1, use a single continuous range with an "
                'inclusive lower bound in one of these forms: "flwr>=X" or '
                '"flwr>=X,<=Y".'
            )

    if lower is None:
        raise ValueError(
            'Invalid "flwr" dependency specifier: an inclusive lower bound is '
            "required for fab_format_version >= 1."
        )

    if upper is not None and upper < lower:
        raise ValueError(
            'Invalid "flwr" dependency specifier: the upper bound must not be '
            "smaller than the lower bound."
        )

    return lower, upper


def _get_flwr_app_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the `[tool.flwr.app]` table without mutating the parsed config."""
    tool = config.get("tool")
    if not isinstance(tool, dict):
        return {}

    flwr = tool.get("flwr")
    if not isinstance(flwr, dict):
        return {}

    app = flwr.get("app")
    return app if isinstance(app, dict) else {}


def _resolve_fab_format_version(app: dict[str, Any]) -> int:
    """Resolve and validate `fab_format_version` from app config."""
    fab_format_version = app.get("fab_format_version", 0)
    if (
        not isinstance(fab_format_version, int)
        or isinstance(fab_format_version, bool)
        or fab_format_version < 0
    ):
        raise ValueError(
            "Invalid [tool.flwr.app].fab_format_version: expected a non-negative "
            "integer."
        )
    return fab_format_version


def _parse_flwr_target_version(app: dict[str, Any]) -> Version | None:
    """Parse optional `flwr_version_target` from app config."""
    target_raw = app.get("flwr_version_target")
    if target_raw is not None and not isinstance(target_raw, str):
        raise ValueError(
            "Invalid [tool.flwr.app].flwr_version_target: expected a string."
        )

    if isinstance(target_raw, str):
        return _parse_version(
            target_raw,
            "[tool.flwr.app].flwr_version_target",
        )
    return None


def _validate_target_within_bounds(
    target_version: Version | None,
    lower: Version | None,
    upper: Version | None,
) -> None:
    """Ensure `flwr_version_target` falls within the derived range, if any."""
    if target_version is None or lower is None:
        return

    if target_version < lower or (upper is not None and target_version > upper):
        raise ValueError(
            "Invalid [tool.flwr.app].flwr_version_target: must fall within the "
            'declared "flwr" dependency range.'
        )


def _build_fab_metadata(
    fab_format_version: int,
    target_version: Version | None,
    lower: Version | None,
    upper: Version | None,
) -> FabFormatMetadata:
    """Create normalized metadata from parsed versions."""
    return FabFormatMetadata(
        fab_format_version=fab_format_version,
        flwr_version_min=str(lower) if lower is not None else None,
        flwr_version_target=str(target_version) if target_version is not None else None,
        flwr_version_max=str(upper) if upper is not None else None,
    )


class _FabFormatV0Rules(_FabFormatRules):
    """Legacy ruleset for fab_format_version 0.

    - `flwr` dependency is optional.
    - `flwr_version_target` is optional.
    - Compatibility bounds are derived only when the declared `flwr` dependency
      can be represented as a single supported range.
    """

    def normalize_and_validate_metadata(
        self, config: dict[str, Any]
    ) -> FabFormatMetadata:
        """Metadata derivation for legacy FABs."""
        app = _get_flwr_app_config(config)
        target_version = _parse_flwr_target_version(app)
        requirement = _get_flwr_requirement(config)
        lower: Version | None = None
        upper: Version | None = None

        if requirement is not None:
            try:
                lower, upper = _derive_flwr_version_bounds(requirement)
            except ValueError:
                lower, upper = None, None

        _validate_target_within_bounds(target_version, lower, upper)
        return _build_fab_metadata(0, target_version, lower, upper)


class _FabFormatV1Rules(_FabFormatRules):
    """Strict ruleset for fab_format_version 1.

    - `flwr` dependency is required.
    - The dependency must include an inclusive lower bound.
    - Optional `flwr_version_target` must fall within the derived range.
    - Unsupported specifier shapes are rejected.
    """

    def normalize_and_validate_metadata(
        self, config: dict[str, Any]
    ) -> FabFormatMetadata:
        """Require and derive strict compatibility metadata for fab_format_version =
        1."""
        app = _get_flwr_app_config(config)
        target_version = _parse_flwr_target_version(app)
        requirement = _get_flwr_requirement(config)
        if requirement is None:
            raise ValueError(
                'Missing "flwr" dependency in [project].dependencies for '
                "fab_format_version >= 1."
            )

        lower, upper = _derive_flwr_version_bounds(requirement)
        _validate_target_within_bounds(target_version, lower, upper)
        return _build_fab_metadata(1, target_version, lower, upper)


_FAB_FORMAT_RULES: dict[int, _FabFormatRules] = {
    0: _FabFormatV0Rules(),
    1: _FabFormatV1Rules(),
}


def _get_fab_format_rules(fab_format_version: int) -> _FabFormatRules:
    """Return the ruleset for a supported `fab_format_version`."""
    if fab_format_version not in _FAB_FORMAT_RULES:
        raise ValueError(
            f"Unsupported [tool.flwr.app].fab_format_version: {fab_format_version}."
        )

    return _FAB_FORMAT_RULES[fab_format_version]


def normalize_and_validate_fab_format(
    config: dict[str, Any],
) -> FabFormatMetadata:
    """Normalize FAB metadata in config and validate `fab_format_version` rules."""
    app = _get_flwr_app_config(config)
    fab_format_version = _resolve_fab_format_version(app)
    return _get_fab_format_rules(fab_format_version).normalize_and_validate_metadata(
        config
    )


def validate_fab_files_for_format(
    config: dict[str, Any], filtered_paths: list[str]
) -> None:
    """Validate the final FAB contents using the selected format ruleset."""
    app = _get_flwr_app_config(config)
    fab_format_version = _resolve_fab_format_version(app)
    _get_fab_format_rules(fab_format_version).validate_fab_contents(
        config, filtered_paths
    )


def get_project_dir(
    fab_id: str,
    fab_version: str,
    fab_hash: str,
) -> Path:
    """Return the project directory based on the given fab_id and fab_version."""
    # Check the fab_id
    if fab_id.count("/") != 1:
        raise ValueError(
            f"Invalid FAB ID: {fab_id}",
        )
    publisher, project_name = fab_id.split("/")
    return (
        get_flwr_home()
        / APP_DIR
        / f"{publisher}.{project_name}.{fab_version}.{fab_hash[:FAB_HASH_TRUNCATION]}"
    )


def get_project_config(project_dir: str | Path) -> dict[str, Any]:
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
    is_valid, errors, _ = validate_fields_in_config(config)
    if not is_valid:
        error_msg = "\n".join([f"  - {error}" for error in errors])
        raise ValueError(
            f"Invalid {FAB_CONFIG_FILE}:\n{error_msg}",
        )

    return config


def fuse_dicts(
    main_dict: T_dict,
    override_dict: T_dict,
    check_keys: bool = True,
) -> T_dict:
    """Merge a config with the overrides.

    If `check_keys` is set to True, an error will be raised if the override
    dictionary contains keys that are not present in the main dictionary.
    Otherwise, only the keys present in the main dictionary will be updated.
    """
    if not isinstance(main_dict, dict) or not isinstance(override_dict, dict):
        raise ValueError("Both dictionaries must be of type dict")

    fused_dict = cast(T_dict, main_dict.copy())

    for key, value in override_dict.items():
        if key in main_dict:
            if isinstance(value, dict):
                fused_dict[key] = fuse_dicts(main_dict[key], value)
            fused_dict[key] = value
        elif check_keys:
            raise ValueError(f"Key '{key}' is not present in the main dictionary")

    return fused_dict


def get_fused_config_from_dir(
    project_dir: Path, override_config: UserConfig
) -> UserConfig:
    """Merge the overrides from a given dict with the config from a Flower App."""
    default_config = get_project_config(project_dir)["tool"]["flwr"]["app"].get(
        "config", {}
    )
    flat_default_config = flatten_dict(default_config)

    return fuse_dicts(flat_default_config, override_config)


def get_fused_config_from_fab(fab_file: Path | bytes, run: Run) -> UserConfig:
    """Fuse default config in a `FAB` with overrides in a `Run`.

    This enables obtaining a run-config without having to install the FAB. This
    function mirrors `get_fused_config_from_dir`. This is useful when the execution
    of the FAB is delegated to a different process.
    """
    default_config = get_fab_config(fab_file)["tool"]["flwr"]["app"].get("config", {})
    flat_config_flat = flatten_dict(default_config)
    return fuse_dicts(flat_config_flat, run.override_config)


def get_fused_config(run: Run) -> UserConfig:
    """Merge the overrides from a `Run` with the config from a FAB.

    Get the config using the fab_id and the fab_version, remove the nesting by adding
    the nested keys as prefixes separated by dots, and fuse it with the override dict.
    """
    # Return empty dict if fab_id or fab_version is empty
    if not run.fab_id or not run.fab_version:
        return {}

    project_dir = get_project_dir(run.fab_id, run.fab_version, run.fab_hash)

    # Return empty dict if project directory does not exist
    if not project_dir.is_dir():
        return {}

    return get_fused_config_from_dir(project_dir, run.override_config)


def flatten_dict(raw_dict: dict[str, Any] | None, parent_key: str = "") -> UserConfig:
    """Flatten dict by joining nested keys with a given separator."""
    if raw_dict is None:
        return {}

    items: list[tuple[str, UserConfigValue]] = []
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


def unflatten_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Unflatten a dict with keys containing separators into a nested dict."""
    unflattened_dict: dict[str, Any] = {}
    separator: str = "."

    for key, value in flat_dict.items():
        parts = key.split(separator)
        d = unflattened_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return unflattened_dict


def parse_config_args(config: list[str] | None, flatten: bool = True) -> dict[str, Any]:
    """Parse separator separated list of key-value pairs separated by '='."""
    overrides: UserConfig = {}

    if config is None:
        return overrides

    # Handle if .toml file is passed
    if len(config) == 1 and config[0].endswith(".toml"):
        with Path(config[0]).expanduser().open("rb") as config_file:
            overrides = flatten_dict(tomli.load(config_file))
        return overrides

    # Regular expression to capture key-value pairs with possible quoted values
    pattern = re.compile(r"(\S+?)=(\'[^\']*\'|\"[^\"]*\"|\S+)")

    flat_overrides = {}
    for config_line in config:
        if config_line:
            # .toml files aren't allowed alongside other configs
            if config_line.endswith(".toml"):
                raise ValueError(
                    "TOML files cannot be passed alongside key-value pairs."
                )

            matches = pattern.findall(config_line)
            toml_str = "\n".join(f"{k} = {v}" for k, v in matches)
            try:
                overrides.update(tomli.loads(toml_str))
                flat_overrides = flatten_dict(overrides) if flatten else overrides
            except tomli.TOMLDecodeError as err:
                raise click.ClickException(
                    "The provided configuration string is in an invalid format. "
                    "The correct format should be, e.g., 'key1=123 key2=false "
                    'key3="string"\', where values must be of type bool, int, '
                    "string, or float. Ensure proper formatting with "
                    "space-separated key-value pairs."
                ) from err

    return flat_overrides


def get_metadata_from_config(config: dict[str, Any]) -> tuple[str, str]:
    """Extract `fab_id` and `fab_version` from a project config."""
    return (
        f"{config['tool']['flwr']['app']['publisher']}/{config['project']['name']}",
        config["project"]["version"],
    )


def user_config_to_configrecord(config: UserConfig) -> ConfigRecord:
    """Construct a `ConfigRecord` out of a `UserConfig`."""
    c_record = ConfigRecord()
    for k, v in config.items():
        c_record[k] = v

    return c_record


def get_fab_config(fab_file: Path | bytes) -> dict[str, Any]:
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
    fab_file_archive: Path | IO[bytes]
    if isinstance(fab_file, bytes):
        fab_file_archive = BytesIO(fab_file)
    elif isinstance(fab_file, Path):
        fab_file_archive = fab_file
    else:
        raise ValueError("fab_file must be either a Path or bytes")

    with zipfile.ZipFile(fab_file_archive, "r") as zipf:
        with zipf.open("pyproject.toml") as file:
            toml_content = file.read().decode("utf-8")
        try:
            conf = tomli.loads(toml_content)
        except tomli.TOMLDecodeError:
            raise ValueError("Invalid TOML content in pyproject.toml") from None

        is_valid, errors, _ = validate_config(conf, check_module=False)
        if not is_valid:
            raise ValueError(errors)

        return conf


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
def validate_fields_in_config(
    config: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
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


def validate_config(
    config: dict[str, Any],
    check_module: bool = True,
    project_dir: str | Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate pyproject.toml."""
    is_valid, errors, warnings = validate_fields_in_config(config)

    if not is_valid:
        return False, errors, warnings

    try:
        normalize_and_validate_fab_format(config)
    except ValueError as err:
        return False, [str(err)], warnings

    # Validate serverapp
    serverapp_ref = config["tool"]["flwr"]["app"]["components"]["serverapp"]
    is_valid, reason = object_ref.validate(serverapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], warnings

    # Validate clientapp
    clientapp_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]
    is_valid, reason = object_ref.validate(clientapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], warnings

    return True, [], warnings
