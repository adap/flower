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
"""Tests for the validation function of name property."""

from pyproject import validate_project_name


def test_valid_name_with_dashes():
    assert validate_project_name(
        "valid-project-name",
        "Expected 'valid-project-name' to be valid",
    )


def test_valid_name_with_underscores():
    assert validate_project_name(
        "valid_project_name"
    ), "Expected 'valid_project_name' to be valid"


def test_invalid_name_with_upper_letters():
    assert not validate_project_name(
        "Invalid Project Name"
    ), "Upper Case and Spaces are not allowed"


def test_name_with_spaces():
    assert not validate_project_name(
        "name with spaces"
    ), "Spaces are not allowed"


def test_empty_name():
    assert not validate_project_name(""), "Empty name is not valid"


def test_long_name():
    # It can be more than 40 but generally 
    # it is recommended not to be more than 40
    long_name = "a" * 41
    assert not validate_project_name(
        long_name
    ), f"Name longer than 40 characters is not recommended"


def test_name_with_special_characters():
    assert not validate_project_name(
        "name!@#"
    ), "Special characters are not allowed"


def test_name_with_special_characters():
    assert not validate_project_name(
        "name!@#"
    ), "Special characters are not allowed"
