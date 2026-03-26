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
"""Tests for FAB format version validation and metadata derivation."""


from typing import Any

import pytest

from .fab_format_version import normalize_and_validate_fab_format


def test_normalize_and_validate_fab_format_rejects_unsupported_version() -> None:
    """Test unsupported fab_format_version values fail explicitly."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 2,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_and_validate_fab_format(config)


def test_normalize_and_validate_fab_format_accepts_target_for_version_zero() -> None:
    """Test flwr_version_target is accepted for fab_format_version=0."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 0,
                    "flwr_version_target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target == "1.27.0"
    assert metadata.flwr_version_max is None


def test_normalize_and_validate_fab_format_derives_bounds_for_version_zero() -> None:
    """Test fab_format_version=0 derives bounds when the flwr dependency is usable."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>=1.26.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 0,
                    "flwr_version_target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target == "1.27.0"
    assert metadata.flwr_version_max == "1.28.0"


def test_v0_fab_format_skips_unsupported_bounds() -> None:
    """Test fab_format_version=0 ignores unrepresentable flwr dependency specifiers."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>1.26.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 0,
                    "flwr_version_target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target == "1.27.0"
    assert metadata.flwr_version_max is None


def test_v1_fab_format_rejects_duplicate_lower_bounds() -> None:
    """Test fab_format_version=1 rejects multiple lower-bound specifiers."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,>=1.27.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="multiple lower bounds"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_duplicate_upper_bounds() -> None:
    """Test fab_format_version=1 rejects multiple upper-bound specifiers."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,<=2.0.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="multiple upper bounds"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_requires_license_file_reference() -> None:
    """Test fab_format_version=1 requires [project].license.file."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match=r"\[project\]\.license"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_inline_license_text() -> None:
    """Test fab_format_version=1 rejects inline license text."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"text": "Apache-2.0"},
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="root-level license file"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_invalid_license_file_name() -> None:
    """Test fab_format_version=1 only allows LICENSE or LICENSE.md."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "legal/LICENSE.txt"},
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab_format_version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match='"LICENSE" or "LICENSE.md"'):
        normalize_and_validate_fab_format(config)
