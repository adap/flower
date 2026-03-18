# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""FAB format version validation and derived metadata helpers."""


from dataclasses import dataclass
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


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
        """Require and derive strict metadata for `fab_format_version = 1`."""
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
