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
"""Tests for legacy default workflows."""


import threading
import time
import unittest
from unittest.mock import Mock, patch

from ..client_manager import SimpleClientManager
from .default_workflows import _update_client_manager


class TestDefaultWorkflow(unittest.TestCase):
    """Tests for default workflows."""

    def test_update_client_manager(self) -> None:
        """Test _update_client_manager function."""
        # Prepare
        sleep = time.sleep
        sleep_patch = patch("time.sleep", lambda x: sleep(x / 100))
        sleep_patch.start()
        expected_node_ids = list(range(100))
        updated_expected_node_ids = list(range(80, 120))
        driver = Mock()
        driver.grpc_driver = Mock()
        driver.run_id = 123
        driver.get_node_ids.return_value = expected_node_ids
        client_manager = SimpleClientManager()
        f_stop = threading.Event()

        # Execute
        client_manager_thread = threading.Thread(
            target=_update_client_manager,
            args=(
                driver,
                client_manager,
                f_stop,
            ),
        )
        client_manager_thread.start()
        # Wait until all nodes are registered via `client_manager.sample()`
        client_manager.sample(len(expected_node_ids))
        # Retrieve all nodes in `client_manager`
        node_ids = {proxy.node_id for proxy in client_manager.all().values()}
        # Update the GetNodesResponse and wait until the `client_manager` is updated
        driver.get_node_ids.return_value = updated_expected_node_ids
        sleep(0.1)
        # Retrieve all nodes in `client_manager`
        updated_node_ids = {proxy.node_id for proxy in client_manager.all().values()}
        # Stop the thread
        f_stop.set()

        # Assert
        assert node_ids == set(expected_node_ids)
        assert updated_node_ids == set(updated_expected_node_ids)

        # Exit
        client_manager_thread.join()
