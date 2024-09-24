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
"""Tests for the validation function of object refs."""

from .object_ref import OBJECT_REF_HELP_STR, validate


def test_validate_object_reference() -> None:
    """Test that validate_object_reference succeeds correctly."""
    # Prepare
    ref = "flwr.cli.run:run"

    # Execute
    is_valid, error = validate(ref)

    # Assert
    assert is_valid
    assert error is None


def test_validate_object_reference_fails() -> None:
    """Test that validate_object_reference fails correctly."""
    # Prepare
    ref = "flwr.cli.run:runa"

    # Execute
    is_valid, error = validate(ref)

    # Assert
    assert not is_valid
    assert (
        error
        == f"Unable to find attribute runa in module flwr.cli.run{OBJECT_REF_HELP_STR}"
    )
