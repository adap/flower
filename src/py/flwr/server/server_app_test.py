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
"""Tests for ServerApp."""


from unittest.mock import MagicMock

import pytest

from flwr.common import Context, RecordSet
from flwr.server import ServerApp, ServerConfig
from flwr.server.driver import Driver


def test_server_app_custom_mode() -> None:
    """Test sampling w/o criterion."""
    # Prepare
    app = ServerApp()
    driver = MagicMock()
    context = Context(state=RecordSet())

    called = {"called": False}

    # pylint: disable=unused-argument
    @app.main()
    def custom_main(driver: Driver, context: Context) -> None:
        called["called"] = True

    # pylint: enable=unused-argument

    # Execute
    app(driver, context)

    # Assert
    assert called["called"]


def test_server_app_exception_when_both_modes() -> None:
    """Test ServerApp error when both compat mode and custom fns are used."""
    # Prepare
    app = ServerApp(config=ServerConfig(num_rounds=3))

    # Execute and assert
    with pytest.raises(ValueError):
        # pylint: disable=unused-argument
        @app.main()
        def custom_main(driver: Driver, context: Context) -> None:
            pass

        # pylint: enable=unused-argument