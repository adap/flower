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
"""Tests for the function that validates name property."""

from .pyproject import validate_project_name


# Happy Flow
def test_valid_name_with_lower_case() -> None:
    """Test a valid single-word project name with all lower case."""
    # Prepare
    name = "myproject"
    expected = True
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, f"Expected {name} to be valid"


def test_valid_name_with_dashes() -> None:
    """Test a valid project name with hyphens inbetween."""
    # Prepare
    name = "valid-project-name"
    expected = True
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, f"Expected {name} to be valid"


def test_valid_name_with_underscores() -> None:
    """Test a valid project name with underscores inbetween."""
    # Prepare
    name = "valid_project_name"
    expected = True
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, f"Expected {name} to be valid"


def test_invalid_name_with_upper_letters() -> None:
    """Tests a project name with Spaces and Uppercase letter."""
    # Prepare
    name = "Invalid Project Name"
    expected = False
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, "Upper Case and Spaces are not allowed"


def test_name_with_spaces() -> None:
    """Tests a project name with spaces inbetween."""
    # Prepare
    name = "name with spaces"
    expected = False
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, "Spaces are not allowed"


def test_empty_name() -> None:
    """Tests use-case for an empty project name."""
    # Prepare
    name = ""
    expected = False
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, "Empty name is not valid"


def test_long_name() -> None:
    """Tests for long project names."""
    # Prepare
    name = "a" * 41
    expected = False
    # Execute
    actual = validate_project_name(name)
    # Assert
    # It can be more than 40 but generally
    # it is recommended not to be more than 40
    assert actual == expected, "Name longer than 40 characters is not recommended"


def test_name_with_special_characters() -> None:
    """Tests for project names with special characters."""
    # Prepare
    name = "name!@#"
    expected = False
    # Execute
    actual = validate_project_name(name)
    # Assert
    assert actual == expected, "Special characters are not allowed"
