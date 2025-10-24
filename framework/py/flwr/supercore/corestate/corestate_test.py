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
"""Tests all CoreState implementations have to conform to."""


import unittest

from . import CoreState


class StateTest(unittest.TestCase):
    """Test all CoreState implementations."""

    # This is to True in each child class
    __test__ = False

    def state_factory(self) -> CoreState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_create_verify_and_delete_token(self) -> None:
        """Test creating, verifying, and deleting tokens."""
        # Prepare
        state = self.state_factory()
        run_id = 42

        # Execute: create a token
        token = state.create_token(run_id)
        assert token is not None

        # Assert: token should be valid
        self.assertTrue(state.verify_token(run_id, token))

        # Execute: delete the token
        state.delete_token(run_id)

        # Assert: token should no longer be valid
        self.assertFalse(state.verify_token(run_id, token))

    def test_create_token_already_exists(self) -> None:
        """Test creating a token that already exists."""
        # Prepare
        state = self.state_factory()
        run_id = 42
        state.create_token(run_id)

        # Execute
        ret = state.create_token(run_id)

        # Assert: The return is None
        self.assertIsNone(ret)

    def test_get_run_id_by_token(self) -> None:
        """Test retrieving run ID by token."""
        # Prepare
        state = self.state_factory()
        run_id = 42
        token = state.create_token(run_id)
        assert token is not None

        # Execute: get run ID by token
        retrieved_run_id1 = state.get_run_id_by_token(token)
        retrieved_run_id2 = state.get_run_id_by_token("nonexistent_token")

        # Assert: should return the correct run ID
        self.assertEqual(retrieved_run_id1, run_id)
        self.assertIsNone(retrieved_run_id2)
