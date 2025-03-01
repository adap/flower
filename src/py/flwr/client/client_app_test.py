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
"""Tests for Flower ClientApp."""


from collections.abc import Iterator
from unittest.mock import Mock

from flwr.common.context import Context

from .client_app import ClientApp


def test_lifecycle_success() -> None:
    """Test the lifecycle decorator with success."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifecycle()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    with app._lifecycle(Mock(spec=Context)):  # pylint: disable=W0212
        pass

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifecycle_failure() -> None:
    """Test the lifecycle decorator with failure."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifecycle()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifecycle(Mock(spec=Context)):  # pylint: disable=W0212
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifecycle_no_yield() -> None:
    """Test the lifecycle decorator with no yield."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()

    @app.lifecycle()
    def test_fn(_: Context) -> Iterator[None]:  # type: ignore
        enter_code()

    # Execute
    try:
        with app._lifecycle(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()


def test_lifecycle_multiple_yields() -> None:
    """Test the lifecycle decorator with multiple yields."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    middle_code = Mock()
    exit_code = Mock()

    @app.lifecycle()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        middle_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifecycle(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    middle_code.assert_called_once()
    exit_code.assert_not_called()
