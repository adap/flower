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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomli

from flwr.common import object_ref


def load_and_validate_with_defaults(
    path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """Load and validate pyproject.toml as dict.

    Returns
    -------
    Tuple[Optional[config], List[str], List[str]]
        A tuple with the optional config in case it exists and is valid
        and associated errors and warnings.
    """
    config = load(path)

    if config is None:
        errors = [
            "Project configuration could not be loaded. `pyproject.toml` does not exist."
        ]
        return (None, errors, [])

    is_valid, errors, warnings = validate(config)

    if not is_valid:
        return (None, errors, warnings)

    # Apply defaults
    defaults = {
        "flower": {
            "engine": {"name": "simulation", "simulation": {"supernode": {"num": 2}}}
        }
    }
    config = apply_defaults(config, defaults)

    return (config, errors, warnings)


def load(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load pyproject.toml and return as dict."""
    if path is None:
        cur_dir = Path.cwd()
        toml_path = cur_dir / "pyproject.toml"
    else:
        toml_path = path

    if not toml_path.is_file():
        return None

    with toml_path.open(encoding="utf-8") as toml_file:
        data = tomli.loads(toml_file.read())
        return data


def validate_fields(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
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

    if "flower" not in config:
        errors.append("Missing [flower] section")
    else:
        if "publisher" not in config["flower"]:
            errors.append('Property "publisher" missing in [flower]')
        if "components" not in config["flower"]:
            errors.append("Missing [flower.components] section")
        else:
            if "serverapp" not in config["flower"]["components"]:
                errors.append('Property "serverapp" missing in [flower.components]')
            if "clientapp" not in config["flower"]["components"]:
                errors.append('Property "clientapp" missing in [flower.components]')

    return len(errors) == 0, errors, warnings


def validate(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate pyproject.toml."""
    is_valid, errors, warnings = validate_fields(config)

    if not is_valid:
        return False, errors, warnings

    # Validate serverapp
    is_valid, reason = object_ref.validate(config["flower"]["components"]["serverapp"])
    if not is_valid and isinstance(reason, str):
        return False, [reason], []

    # Validate clientapp
    is_valid, reason = object_ref.validate(config["flower"]["components"]["clientapp"])

    if not is_valid and isinstance(reason, str):
        return False, [reason], []

    return True, [], []


def apply_defaults(
    config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply defaults to config."""
    for key in defaults:
        if key in config:
            if isinstance(config[key], dict) and isinstance(defaults[key], dict):
                apply_defaults(config[key], defaults[key])
        else:
            config[key] = defaults[key]
    return config
