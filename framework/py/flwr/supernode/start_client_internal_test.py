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
"""Tests for the Main Loop of Flower SuperNode."""


import unittest
from unittest.mock import Mock, patch

from flwr.common import Context
from flwr.common.typing import Fab

from .start_client_internal import _pull_and_store_message


class TestStartClientInternal(unittest.TestCase):
    """Test cases for the main loop."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.mock_state = Mock()
        self.node_id = 111
        self.mock_state.get_node_id.return_value = self.node_id
        self.mock_ffs = Mock()
        self.mock_object_store = Mock()
        self.mock_receive = Mock()
        self.mock_get_run = Mock()
        self.mock_get_fab = Mock()

    def test_pull_and_store_message_no_message(self) -> None:
        """Test that no message is pulled when there are no messages."""
        # Prepare
        self.mock_receive.return_value = None

        # Execute
        res = _pull_and_store_message(
            state=self.mock_state,
            ffs=self.mock_ffs,
            object_store=self.mock_object_store,
            node_config={},  # No need for this test
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
        )

        # Assert
        assert res is None
        self.mock_receive.assert_called_once()
        self.mock_get_run.assert_not_called()
        self.mock_get_fab.assert_not_called()

    def test_pull_and_store_message_with_known_run_id(self) -> None:
        """Test that a message of a known run ID is pulled and stored."""
        # Prepare
        message = Mock()
        message.metadata = Mock(
            run_id=110,
            group_id="test_group",
            message_type="query",
            message_id="mock_message_id",
        )
        self.mock_receive.return_value = message
        self.mock_state.get_run.return_value = Mock()  # Mock non-None return

        # Execute
        res = _pull_and_store_message(
            state=self.mock_state,
            ffs=self.mock_ffs,
            object_store=self.mock_object_store,
            node_config={},  # No need for this test
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
        )

        # Assert: the run ID should be returned
        assert res == 110

        # Assert: the message should be stored
        self.mock_state.get_run.assert_called_once()
        self.mock_state.store_message.assert_called_once_with(message)

        # Assert: receive should be called once, while other methods must not be called
        self.mock_receive.assert_called_once()
        self.mock_get_run.assert_not_called()
        self.mock_get_fab.assert_not_called()

    def test_pull_and_store_message_with_unknown_run_id(self) -> None:
        """Test that a message of an unknown run ID is pulled and stored."""
        # Prepare: Mock connection methods
        mock_msg = Mock()
        run_id = 999
        mock_msg.metadata = Mock(
            run_id=run_id,
            group_id="test_group",
            message_type="query",
            message_id="mock_message_id",
        )
        fab = Fab(
            hash_str="abc123",
            content=b"test_fab_content",
        )
        mock_run = Mock(
            run_id=run_id,
            fab_hash=fab.hash_str,
            override_config={},
        )
        self.mock_receive.return_value = mock_msg
        self.mock_get_run.return_value = mock_run
        self.mock_get_fab.return_value = fab
        self.mock_state.get_run.return_value = None

        # Prepare: Mock the get_fused_config_from_fab return
        mock_fused_run_config = {"mock_key": "mock_value"}

        # Execute
        with patch(
            "flwr.supernode.start_client_internal.get_fused_config_from_fab"
        ) as mock_get_fused_config:
            mock_get_fused_config.return_value = mock_fused_run_config
            res = _pull_and_store_message(
                state=self.mock_state,
                ffs=self.mock_ffs,
                object_store=self.mock_object_store,
                node_config={},  # No need for this test
                receive=self.mock_receive,
                get_run=self.mock_get_run,
                get_fab=self.mock_get_fab,
            )

        # Assert: the run ID should be returned
        assert res == run_id

        # Assert: the message should be stored
        self.mock_state.store_message.assert_called_once_with(mock_msg)

        # Assert: the Run and FAB should be fetched and stored
        self.mock_get_run.assert_called_once_with(run_id)
        self.mock_get_fab.assert_called_once_with(fab.hash_str, run_id)
        self.mock_ffs.put.assert_called_once_with(fab.content, {})
        self.mock_state.store_run.assert_called_once_with(mock_run)

        # Assert: the Context should be created and stored
        self.mock_state.store_context.assert_called_once()
        args, _ = self.mock_state.store_context.call_args
        ctxt = args[0]
        assert isinstance(ctxt, Context)
        assert ctxt.run_id == run_id
        assert ctxt.node_id == self.node_id
        assert ctxt.node_config == {}
        assert ctxt.run_config == mock_fused_run_config
