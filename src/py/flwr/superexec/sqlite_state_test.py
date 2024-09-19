"""Test cases for the sqlite state."""

import json
import unittest

from flwr.common.typing import UserConfig

from .sqlite_state import SqliteExecState


class TestSqliteExecState(unittest.TestCase):
    """Test the sqlite state."""

    def setUp(self) -> None:
        """Set up an in-memory SQLite database for testing."""
        self.db_path = ":memory:"
        self.exec_state = SqliteExecState(self.db_path)

    def test_store_run(self) -> None:
        """Test storing a run in the database."""
        run_id = 1
        run_config = UserConfig({"param1": "value1", "param2": "value2"})
        fab_hash = "abc123"

        self.exec_state.store_run(run_id, run_config, fab_hash)

        # Verify that the run was inserted into the database
        cursor = self.exec_state.conn.cursor()
        cursor.execute(
            "SELECT run_id, run_config, fab_hash FROM runs WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], run_id)
        self.assertEqual(json.loads(row[1]), run_config)
        self.assertEqual(row[2], fab_hash)

    def test_get_run_config(self) -> None:
        """Test retrieving the run config for a specific run_id."""
        run_id = 1
        run_config = UserConfig({"param1": "value1", "param2": "value2"})
        fab_hash = "abc123"

        # First, store the run
        self.exec_state.store_run(run_id, run_config, fab_hash)

        # Now, retrieve the run config
        retrieved_config = self.exec_state.get_run_config(run_id)

        self.assertIsNotNone(retrieved_config)
        self.assertEqual(retrieved_config, run_config)

    def test_get_run_config_nonexistent(self) -> None:
        """Test retrieving the run config for a non-existent run_id."""
        run_id = 999
        retrieved_config = self.exec_state.get_run_config(run_id)
        self.assertIsNone(retrieved_config)

    def test_get_fab_hash(self) -> None:
        """Test retrieving the fab_hash for a specific run_id."""
        run_id = 1
        run_config = UserConfig({"param1": "value1", "param2": "value2"})
        fab_hash = "abc123"

        # First, store the run
        self.exec_state.store_run(run_id, run_config, fab_hash)

        # Now, retrieve the fab_hash
        retrieved_fab_hash = self.exec_state.get_fab_hash(run_id)

        self.assertIsNotNone(retrieved_fab_hash)
        self.assertEqual(retrieved_fab_hash, fab_hash)

    def test_get_fab_hash_nonexistent(self) -> None:
        """Test retrieving the fab_hash for a non-existent run_id."""
        run_id = 999
        retrieved_fab_hash = self.exec_state.get_fab_hash(run_id)
        self.assertIsNone(retrieved_fab_hash)

    def test_get_runs(self) -> None:
        """Test retrieving all run IDs from the database."""
        run_config_1 = UserConfig({"param1": "value1"})
        run_config_2 = UserConfig({"param2": "value2"})

        self.exec_state.store_run(1, run_config_1, "hash1")
        self.exec_state.store_run(2, run_config_2, "hash2")

        run_ids = self.exec_state.get_runs()

        self.assertEqual(run_ids, [1, 2])

    def test_get_runs_empty(self) -> None:
        """Test retrieving runs when no runs are stored."""
        run_ids = self.exec_state.get_runs()
        self.assertEqual(run_ids, [])


if __name__ == "__main__":
    unittest.main(verbosity=3)
