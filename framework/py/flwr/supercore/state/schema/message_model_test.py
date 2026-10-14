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
"""Tests for destination-oriented message storage constraints."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from sqlalchemy import Connection, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError

from flwr.supercore.state.alembic.utils import build_alembic_config
from flwr.supercore.state.schema.corestate_models import FlwrBase


@pytest.fixture(name="message_connection")
def message_connection_fixture() -> Iterator[Connection]:
    """Create an isolated message table containing one pending message."""
    engine = create_engine("sqlite:///:memory:")
    table = FlwrBase.metadata.tables["message"]
    table.create(engine)
    created_at = datetime(2026, 10, 14, tzinfo=UTC)
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON")
        conn.execute(
            table.insert().values(
                message_id="parent",
                run_id=2**64 - 1,
                src_node_id=2**63,
                dst_node_id=2**63 + 1,
                src_task_id=2**63 + 2,
                dst_task_id=None,
                group_id="",
                created_at=created_at.timestamp(),
                ttl=60.0,
                message_type="train",
                expires_at=created_at + timedelta(seconds=60),
            )
        )
        yield conn
    engine.dispose()


def test_message_retains_metadata_and_supports_multiple_replies(
    message_connection: Connection,
) -> None:
    """Retain correlated messages independently of the parent's closing state."""
    table = FlwrBase.metadata.tables["message"]
    conn = message_connection
    parent = dict(conn.execute(select(table)).mappings().one())
    assert parent["run_id"] == 2**64 - 1
    assert parent["src_node_id"] == 2**63
    assert parent["dst_node_id"] == 2**63 + 1
    assert parent["src_task_id"] == 2**63 + 2
    assert parent["dst_task_id"] is None
    assert parent["cleared"] is False
    assert parent["reply_to_message_id"] is None

    for message_id in ("reply-1", "reply-2"):
        conn.execute(
            table.insert().values(
                **{
                    **parent,
                    "message_id": message_id,
                    "reply_to_message_id": "parent",
                    "ttl": 120.0,
                    "expires_at": parent["expires_at"] + timedelta(seconds=60),
                }
            )
        )
    conn.execute(
        table.update()
        .where(table.c.message_id == "parent")
        .values(cleared=True, acknowledged_at=datetime(2026, 10, 14, tzinfo=UTC))
    )
    assert conn.execute(
        select(table.c.message_id)
        .where(table.c.cleared.is_(False))
        .order_by(table.c.message_id)
    ).scalars().all() == ["reply-1", "reply-2"]
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        conn.execute(table.delete().where(table.c.message_id == "parent"))


def test_message_rejects_missing_correlation(message_connection: Connection) -> None:
    """Require a non-null correlation ID to reference retained metadata."""
    table = FlwrBase.metadata.tables["message"]
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        message_connection.execute(table.update().values(reply_to_message_id="missing"))


def test_message_migration_downgrades_correlated_rows(
    message_connection: Connection,
) -> None:
    """Allow migration rollback with correlated rows and SQLite FKs enabled."""
    table = FlwrBase.metadata.tables["message"]
    parent = dict(message_connection.execute(select(table)).mappings().one())
    engine = create_engine("sqlite:///:memory:")
    config = build_alembic_config(engine)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON")
        conn.commit()
        config.attributes["connection"] = conn
        command.upgrade(config, "6e5225b231fc")
        conn.execute(table.insert().values(**parent))
        conn.execute(
            table.insert().values(
                **{**parent, "message_id": "reply", "reply_to_message_id": "parent"}
            )
        )
        conn.commit()
        command.downgrade(config, "8f4785e2dc47")
        assert not inspect(conn).has_table("message")
        assert inspect(conn).has_table("message_ins")
        command.upgrade(config, "6e5225b231fc")
        assert inspect(conn).has_table("message")
    engine.dispose()


@pytest.mark.parametrize(
    ("cleared", "acknowledged", "cancelled", "valid"),
    [
        (True, False, False, True),
        (True, True, False, True),
        (True, False, True, True),
        (True, True, True, False),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_message_enforces_closing_state(
    message_connection: Connection,
    cleared: bool,
    acknowledged: bool,
    cancelled: bool,
    valid: bool,
) -> None:
    """Allow each closing outcome and reject inconsistent terminal timestamps."""
    timestamp = datetime(2026, 10, 14, tzinfo=UTC)
    stmt = (
        FlwrBase.metadata.tables["message"]
        .update()
        .values(
            cleared=cleared,
            acknowledged_at=timestamp if acknowledged else None,
            cancelled_at=timestamp if cancelled else None,
        )
    )
    if valid:
        message_connection.execute(stmt)
    else:
        with pytest.raises(IntegrityError, match="CHECK constraint"):
            message_connection.execute(stmt)
