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
"""Tests LinkStateFactory."""


import tempfile
import unittest
from unittest.mock import patch

from flwr.server.superlink.linkstate import (
    InMemoryLinkState,
    LinkStateFactory,
    SqliteLinkState,
)
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.superlink.federation import NoOpFederationManager


class LinkStateFactoryTest(unittest.TestCase):
    """Test LinkStateFactory singleton pattern."""

    def test_linkstate_factory_singleton_inmemory(self) -> None:
        """Test that LinkStateFactory returns the same InMemoryLinkState instance."""
        # Prepare
        factory = LinkStateFactory(FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager())

        # Execute
        state1 = factory.state()
        state2 = factory.state()
        state3 = factory.state()

        # Assert
        self.assertIsInstance(state1, InMemoryLinkState)
        self.assertIs(state1, state2)
        self.assertIs(state2, state3)

    def test_linkstate_factory_singleton_sqlite_file(self) -> None:
        """Test that LinkStateFactory returns the same SqliteLinkState instance."""
        # Prepare
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        factory = LinkStateFactory(db_path, NoOpFederationManager())

        # Execute
        state1 = factory.state()
        state2 = factory.state()
        state3 = factory.state()

        # Assert
        self.assertIsInstance(state1, SqliteLinkState)
        self.assertIs(state1, state2)
        self.assertIs(state2, state3)

        # Cleanup
        import os

        os.unlink(db_path)

    def test_linkstate_factory_singleton_sqlite_memory(self) -> None:
        """Test that SQLite :memory: database also uses singleton pattern."""
        # Prepare
        factory = LinkStateFactory(":memory:", NoOpFederationManager())

        # Execute
        state1 = factory.state()
        state2 = factory.state()
        state3 = factory.state()

        # Assert
        self.assertIsInstance(state1, SqliteLinkState)
        self.assertIs(state1, state2)
        self.assertIs(state2, state3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
