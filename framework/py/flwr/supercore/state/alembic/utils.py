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
"""Helpers for running and validating Alembic migrations."""


from logging import DEBUG, INFO
from pathlib import Path
from tempfile import TemporaryDirectory

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

BASELINE_REVISION = "8e65d8ae60b0"
ALEMBIC_VERSION_TABLE = "alembic_version"


def get_combined_metadata() -> MetaData:
    """Combine all Flower state metadata objects into a single MetaData instance.

    This ensures Alembic can track all tables across CoreState, LinkState, and
    ObjectStore.

    Returns
    -------
    MetaData
        Combined SQLAlchemy MetaData with all Flower state tables.
    """
    # Start with linkstate tables
    metadata = create_linkstate_metadata()

    # Add corestate tables
    corestate_metadata = create_corestate_metadata()
    for table in corestate_metadata.tables.values():
        table.to_metadata(metadata)

    # Add objectstore tables
    objectstore_metadata = create_objectstore_metadata()
    for table in objectstore_metadata.tables.values():
        table.to_metadata(metadata)

    return metadata


def _build_alembic_config(engine: Engine) -> Config:
    """Create Alembic config with script location and DB URL."""
    alembic_dir = Path(__file__).resolve().parent
    config = Config()
    config.set_main_option("script_location", str(alembic_dir))
    config.set_main_option("sqlalchemy.url", str(engine.url))
    return config


def _get_user_table_names(engine: Engine) -> set[str]:
    """Return non-internal table names for the given engine."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    # Exclude SQLite internal tables (for example, sqlite_sequence)
    return {name for name in table_names if not name.startswith("sqlite_")}


def _has_alembic_version_table(engine: Engine) -> bool:
    """Return True if the Alembic version table exists."""
    inspector = inspect(engine)
    return inspector.has_table(ALEMBIC_VERSION_TABLE)


def _get_baseline_revision(engine: Engine) -> str:
    """Return the revision ID of the first migration (no down_revision).

    Falls back to BASELINE_REVISION constant if unable to determine dynamically.
    """
    try:
        script = ScriptDirectory.from_config(_build_alembic_config(engine))
        for rev in script.walk_revisions():
            if rev.down_revision is None:
                return rev.revision
    except (OSError, ValueError, RuntimeError):
        # Failed to read script directory or parse migrations
        pass
    return BASELINE_REVISION


def _verify_legacy_schema_matches_baseline(
    engine: Engine, inspector: Inspector | None = None
) -> tuple[bool, str]:
    """Verify legacy schema matches baseline tables and columns.

    Only missing tables/columns are reported as errors.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message). If valid, error_message is empty.
    """
    if inspector is None:
        inspector = inspect(engine)

    expected_metadata = get_combined_metadata()
    existing_tables = set(inspector.get_table_names())

    # Filter out SQLite internal tables
    existing_tables = {t for t in existing_tables if not t.startswith("sqlite_")}

    expected_tables = set(expected_metadata.tables.keys())
    missing_tables = expected_tables - existing_tables

    if missing_tables:
        table_list = ", ".join(sorted(existing_tables))
        missing_str = ", ".join(sorted(missing_tables))
        return False, (
            f"Detected tables: [{table_list}]. "
            f"Missing baseline tables: [{missing_str}]."
        )

    # Verify columns for each expected table
    for table_name, table in expected_metadata.tables.items():
        expected_columns = {col.name for col in table.columns}
        actual_columns = {col["name"] for col in inspector.get_columns(table_name)}

        missing_cols = expected_columns - actual_columns
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            return False, (
                f"Table '{table_name}' missing columns: [{missing_cols_str}]."
            )

    return True, ""


def stamp_existing_database(engine: Engine, revision: str = BASELINE_REVISION) -> None:
    """Stamp an existing legacy database to the baseline Alembic revision."""
    command.stamp(_build_alembic_config(engine), revision)


def run_migrations(engine: Engine) -> None:
    """Run pending Alembic migrations, handling pre-Alembic legacy databases.

    Expected scenarios:
    - If alembic_version exists: run migrations normally.
    - If DB is empty, e.g. when newly created: run migrations normally.
    - If DB is pre-Alembic and schema is mismatched: exit with guidance.
    - If DB is pre-Alembic and schema matches baseline: stamp, then upgrade.
    """
    config = _build_alembic_config(engine)
    has_version_table = _has_alembic_version_table(engine)

    # Standard database with version tracking: just upgrade.
    if has_version_table:
        command.upgrade(config, "head")
        return

    table_names = _get_user_table_names(engine)

    # Empty/new database: run all migrations from scratch.
    if not table_names:
        command.upgrade(config, "head")
        return

    # Pre-Alembic database detected: verify baseline schema before stamping.
    baseline_revision = _get_baseline_revision(engine)
    inspector = inspect(engine)
    log(DEBUG, "verifying legacy db")
    is_valid, error_msg = _verify_legacy_schema_matches_baseline(engine, inspector)

    # This is an edge case and unlikely to happen since SuperLink requires a specific
    # schema to operate normally.
    if not is_valid:
        flwr_exit(
            ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH,
            "Detected a pre-Alembic Flower state database, but its schema does not "
            f"match the baseline migration (revision {baseline_revision}). "
            "Back up the database and either migrate it manually to the baseline "
            "schema or start with a fresh database. "
            f"{error_msg}",
        )

    log(
        INFO,
        "Detected pre-Alembic state database without alembic_version; stamping to %s "
        "before upgrading.",
        baseline_revision,
    )
    stamp_existing_database(engine, baseline_revision)
    command.upgrade(config, "head")


def get_current_revision(engine: Engine) -> str | None:
    """Return the current Alembic revision for the given database."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        return context.get_current_revision()


def check_migrations_pending(engine: Engine) -> bool:
    """Return True if the database is not on the latest migration head."""
    current = get_current_revision(engine)
    script = ScriptDirectory.from_config(_build_alembic_config(engine))
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
