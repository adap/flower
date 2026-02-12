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
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, inspect
from sqlalchemy.engine import Engine

from flwr.common.exit import ExitCode
from flwr.supercore.state.alembic.utils import (
    ALEMBIC_VERSION_TABLE,
    _get_baseline_metadata,
    _metadata_providers,
    build_alembic_config,
    get_combined_metadata,
    register_metadata_provider,
    run_migrations,
)


class TestAlembicRun(unittest.TestCase):
    """Test Alembic migration helper utilities."""

    def setUp(self) -> None:
        """Create temporary directory for test databases."""
        self.temp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def create_engine(self, db_name: str = "state.db") -> Engine:
        """Create a SQLAlchemy engine for a test database."""
        db_path = self.temp_path / db_name
        return create_engine(f"sqlite:///{db_path}")

    def test_run_migrations_sets_revision(self) -> None:
        """Ensure migrations advance the database to the latest head."""
        # Prepare
        engine = self.create_engine()
        try:
            # Execute & Assert
            # Initially, there should be no alembic_version table or revision.
            self.assertIsNone(get_current_revision(engine))
            self.assertTrue(check_migrations_pending(engine))

            run_migrations(engine)

            # After migration, alembic_version should be set to the latest head.
            current = get_current_revision(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            self.assertIn(current, script.get_heads())
            # No pending migrations should remain.
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()

    def test_migrated_schema_matches_metadata(self) -> None:
        """Verify that migrations match current SQLAlchemy metadata."""
        # Prepare
        metadata = get_combined_metadata()
        engine = self.create_engine()
        try:
            # Execute: create a fresh database and run migrations
            run_migrations(engine)
            with engine.connect() as connection:
                context = MigrationContext.configure(
                    connection,
                    opts={
                        "compare_type": True,
                        "compare_server_default": True,
                    },
                )
                # Compare the migrated database schema against the metadata
                diffs = compare_metadata(context, metadata)
            # Assert
            self.assertEqual(diffs, [])
        finally:
            engine.dispose()

    def test_legacy_database_is_stamped_and_upgraded_successfully(self) -> None:
        """Ensure legacy databases without alembic_version is stamped and upgraded."""
        # Prepare
        engine = self.create_engine()
        try:
            # Execute & Assert
            # Simulate pre-Alembic behavior: create tables at baseline schema. By
            # construction, there is no alembic_version table or revision.
            baseline_metadata = _get_baseline_metadata()
            baseline_metadata.create_all(engine)
            self.assertIsNone(get_current_revision(engine))
            self.assertFalse(inspect(engine).has_table(ALEMBIC_VERSION_TABLE))

            run_migrations(engine)

            # After migration, alembic_version should be set to the latest head with
            # no pending migrations.
            current = get_current_revision(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            self.assertIn(current, script.get_heads())
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()

    def test_legacy_mismatch_with_missing_tables_exits_with_guidance(self) -> None:
        """Ensure mismatched legacy schemas should fail and exit with a clear error."""
        # Prepare
        engine = self.create_engine()
        try:
            # Create only a subset of baseline tables to simulate incomplete legacy DB
            # Note that this is unlikely to happen as SuperLink requires a specific
            # schema, but we test it for completeness.
            baseline_metadata = _get_baseline_metadata()
            tables_to_create = list(baseline_metadata.tables.values())[:3]
            for table in tables_to_create:
                table.create(engine)

            # Execute & Assert
            with patch("flwr.supercore.state.alembic.utils.flwr_exit") as mock_exit:
                run_migrations(engine)

                # Verify flwr_exit was called with correct arguments
                mock_exit.assert_called_once()
                call_args = mock_exit.call_args
                self.assertEqual(
                    call_args[0][0], ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH
                )
                # Verify error message mentions missing baseline tables
                error_msg = call_args[0][1]
                self.assertIn("missing baseline tables", error_msg.lower())
        finally:
            engine.dispose()

    def test_legacy_mismatch_with_missing_columns_exits_with_guidance(self) -> None:
        """Ensure legacy schemas with missing columns should fail and exit with a clear
        error."""
        # Prepare
        engine = self.create_engine()
        try:
            # Create node table with only some columns (missing required ones)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE node (node_id INTEGER, status TEXT)"
                )
            # Create other tables with baseline schemas
            baseline_metadata = _get_baseline_metadata()
            for table_name in baseline_metadata.tables:
                if table_name != "node":
                    baseline_metadata.tables[table_name].create(engine)

            # Execute & Assert
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

    def test_legacy_database_with_extra_tables_and_columns_succeeds(self) -> None:
        """Ensure legacy databases with extra tables/columns can be migrated.

        This tests backward compatibility: a legacy DB might have extra tables or
        columns that were added manually. The verification should be permissive
        and only fail on MISSING baseline tables/columns.
        """
        # Prepare
        engine = self.create_engine()
        try:
            # Create baseline schema
            baseline_metadata = _get_baseline_metadata()
            baseline_metadata.create_all(engine)

            # Commit the transaction to flush tables
            engine.dispose()
            engine = self.create_engine()

            # Add extra table and column to simulate forward-compatible scenario
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE custom_user_table (id INTEGER)"
                )
                # Add extra column to existing table
                inspector = inspect(engine)
                if inspector.has_table("node"):
                    connection.exec_driver_sql(
                        "ALTER TABLE node ADD COLUMN custom_field TEXT"
                    )

            # Execute: should succeed and stamp/upgrade successfully
            run_migrations(engine)

            current = get_current_revision(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            # Assert
            self.assertIn(current, script.get_heads())
            self.assertFalse(check_migrations_pending(engine))
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


class TestMetadataProviderRegistry(unittest.TestCase):
    """Test external metadata provider registration and collision detection."""

    def setUp(self) -> None:
        """Save the current state of the metadata providers registry."""
        self.original_providers = _metadata_providers.copy()

    def tearDown(self) -> None:
        """Restore the metadata providers registry to its original state."""
        _metadata_providers.clear()
        _metadata_providers.extend(self.original_providers)

    def test_external_provider_table_is_included_in_combined_metadata(self) -> None:
        """Ensure external provider tables are included in combined metadata."""

        # Prepare: Define a mock external metadata provider with a unique table
        def mock_external_provider() -> MetaData:
            metadata = MetaData()
            Table(
                "external_custom_table",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("name", String),
            )
            return metadata

        # Execute: Register the provider
        register_metadata_provider(mock_external_provider)

        # Assert: The external table should be in the combined metadata
        combined = get_combined_metadata()
        self.assertIn("external_custom_table", combined.tables)
        # Verify the table has expected columns
        table = combined.tables["external_custom_table"]
        column_names = {col.name for col in table.columns}
        self.assertEqual(column_names, {"id", "name"})

    def test_external_provider_collision_raises_error(self) -> None:
        """Ensure table name collisions from external providers raise ValueError."""

        # Prepare: Define a mock provider that collides with existing 'node' table
        # (node is defined in linkstate_tables.py)
        def mock_colliding_provider() -> MetaData:
            metadata = MetaData()
            Table(
                "node",  # This table already exists in linkstate_tables
                metadata,
                Column("id", Integer, primary_key=True),
            )
            return metadata

        # Execute: Register the colliding provider
        register_metadata_provider(mock_colliding_provider)

        # Assert: Getting combined metadata should raise ValueError
        with self.assertRaises(ValueError) as context:
            get_combined_metadata()

        # Verify error message identifies the collision
        error_msg = str(context.exception)
        self.assertIn("node", error_msg)
        self.assertIn("collision", error_msg.lower())
        self.assertIn("mock_colliding_provider", error_msg)
