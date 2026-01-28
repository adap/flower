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
"""Tests for Alembic migration helpers."""


import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

from flwr.common.exit import ExitCode
from flwr.supercore.state.alembic.utils import (
    ALEMBIC_VERSION_TABLE,
    BASELINE_REVISION,
    build_alembic_config,
    get_baseline_revision,
    get_combined_metadata,
    run_migrations,
)


class TestAlembicRun(unittest.TestCase):
    """Test Alembic migration helper utilities."""

    def test_run_migrations_sets_revision(self) -> None:
        """Ensure migrations advance the database to the latest head."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                self.assertIsNone(get_current_revision(engine))
                self.assertTrue(check_migrations_pending(engine))

                run_migrations(engine)

                current = get_current_revision(engine)
                script = ScriptDirectory.from_config(build_alembic_config(engine))
                self.assertIn(current, script.get_heads())
                self.assertFalse(check_migrations_pending(engine))
            finally:
                engine.dispose()

    def test_migrated_schema_matches_metadata(self) -> None:
        """Verify that migrations match current SQLAlchemy metadata."""
        metadata = get_combined_metadata()
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                run_migrations(engine)
                with engine.connect() as connection:
                    context = MigrationContext.configure(
                        connection,
                        opts={
                            "compare_type": True,
                            "compare_server_default": True,
                        },
                    )
                    diffs = compare_metadata(context, metadata)
                self.assertEqual(diffs, [])
            finally:
                engine.dispose()

    def test_legacy_database_is_stamped_and_upgraded_successfully(self) -> None:
        """Ensure legacy databases without alembic_version is stamped and upgraded."""
        metadata = get_combined_metadata()
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                # Simulate pre-Alembic behavior: create tables directly.
                metadata.create_all(engine)
                self.assertIsNone(get_current_revision(engine))
                self.assertFalse(inspect(engine).has_table(ALEMBIC_VERSION_TABLE))

                run_migrations(engine)

                current = get_current_revision(engine)
                script = ScriptDirectory.from_config(build_alembic_config(engine))
                self.assertIn(current, script.get_heads())
                self.assertFalse(check_migrations_pending(engine))
            finally:
                engine.dispose()

    def test_legacy_mismatch_raises_with_guidance(self) -> None:
        """Ensure mismatched legacy schemas should fail with a clear error."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                # Create a subset of tables to trigger verification failure.
                with engine.begin() as connection:
                    connection.exec_driver_sql("CREATE TABLE legacy_only (id INTEGER)")

                with patch("flwr.supercore.state.alembic.utils.flwr_exit") as mock_exit:
                    run_migrations(engine)

                    # Verify flwr_exit was called with correct arguments
                    mock_exit.assert_called_once()
                    call_args = mock_exit.call_args
                    self.assertEqual(
                        call_args[0][0], ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH
                    )

            finally:
                engine.dispose()

    def test_legacy_mismatch_with_missing_columns_raises(self) -> None:
        """Ensure legacy schemas with missing columns fail verification."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                # Create node table with only some columns (missing required ones)
                with engine.begin() as connection:
                    connection.exec_driver_sql(
                        "CREATE TABLE node (node_id INTEGER, status TEXT)"
                    )
                    # Create other tables with complete schemas
                    metadata = get_combined_metadata()
                    for table_name in metadata.tables:
                        if table_name != "node":
                            metadata.tables[table_name].create(engine)

                with patch("flwr.supercore.state.alembic.utils.flwr_exit") as mock_exit:
                    run_migrations(engine)

                    # Verify flwr_exit was called
                    mock_exit.assert_called_once()
                    call_args = mock_exit.call_args
                    self.assertEqual(
                        call_args[0][0], ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH
                    )
                    # Verify error message mentions missing columns
                    error_msg = call_args[0][1]
                    self.assertIn("missing columns", error_msg.lower())

            finally:
                engine.dispose()

    def test_check_migrations_in_sync(self) -> None:
        """Ensure migrations are in sync with metadata."""
        self.assertTrue(check_migrations_in_sync())

    def testget_baseline_revision_returns_correct_value(self) -> None:
        """Ensure get_baseline_revision dynamically finds the baseline."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                baseline = get_baseline_revision()
                # Should match the constant (or be dynamically discovered)
                self.assertEqual(baseline, BASELINE_REVISION)
            finally:
                engine.dispose()


def get_current_revision(engine: Engine) -> str | None:
    """Return the current Alembic revision for the given database."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        return context.get_current_revision()


def check_migrations_pending(engine: Engine) -> bool:
    """Return True if the database is not on the latest migration head."""
    current = get_current_revision(engine)
    script = ScriptDirectory.from_config(build_alembic_config(engine))
    heads = set(script.get_heads())
    if current is None:
        return True
    return current not in heads


def check_migrations_in_sync() -> bool:
    """Return True if migrations produce no diff versus current metadata."""
    metadata = get_combined_metadata()
    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "state.db"
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            with engine.connect() as connection:
                context = MigrationContext.configure(
                    connection,
                    opts={
                        "compare_type": True,
                        "compare_server_default": True,
                    },
                )
                diffs = compare_metadata(context, metadata)
        finally:
            engine.dispose()
    return len(diffs) == 0
