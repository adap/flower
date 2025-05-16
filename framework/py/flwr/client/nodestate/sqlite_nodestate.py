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
"""SQLite-based implementation of NodeState."""


from typing import Optional

from flwr.common.sqlite_state_mixin import SqliteStateMixin
from flwr.common.state_utils import convert_sint64_to_uint64, convert_uint64_to_sint64

from .nodestate import NodeState

SQL_CREATE_TABLE_META = """
CREATE TABLE IF NOT EXISTS meta (
    id              TEXT PRIMARY KEY CHECK (id = 'singleton'),
    node_id         INTEGER
);
"""


class SqliteNodeState(NodeState, SqliteStateMixin):
    """SQLite-based NodeState implementation."""

    @property
    def schema_setup_commands(self) -> list[str]:
        """Return the schema setup commands."""
        return [
            SQL_CREATE_TABLE_META,
        ]

    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Create tables if they don't exist yet."""
        ret = super().initialize(log_queries)

        # Initialize the meta
        query = "INSERT INTO meta (id) VALUES (?) ON CONFLICT(id) DO NOTHING;"
        self.query(query, ("singleton",))
        return ret

    def set_node_id(self, node_id: Optional[int]) -> None:
        """Set the node ID."""
        sint_node_id = (
            convert_uint64_to_sint64(node_id) if node_id is not None else None
        )
        query = "UPDATE meta SET node_id = ? WHERE id = 'singleton';"
        self.query(query, (sint_node_id,))

    def get_node_id(self) -> int:
        """Get the node ID."""
        query = "SELECT node_id FROM meta WHERE id = 'singleton';"
        result = self.query(query)
        if result[0]["node_id"] is None:
            raise ValueError("Node ID not set")
        return convert_sint64_to_uint64(result[0]["node_id"])
