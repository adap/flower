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
"""Tests for utility functions for the `start_driver`."""


import time
import unittest
from threading import Event
from typing import Optional
from unittest.mock import Mock, patch

from ..client_manager import SimpleClientManager
from .app_utils import start_update_client_manager_thread


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_start_update_client_manager_thread(self) -> None:
        """Test start_update_client_manager_thread function."""
        # Prepare
        expected_node_ids = list(range(100))
        updated_expected_node_ids = list(range(80, 120))
        driver = Mock()
        driver.grpc_driver = Mock()
        driver.run_id = 123
        driver.get_node_ids.return_value = expected_node_ids
        client_manager = SimpleClientManager()
        original_wait = Event.wait

        def custom_wait(self: Event, timeout: Optional[float] = None) -> None:
            if timeout is not None:
                timeout /= 100
            original_wait(self, timeout)

        # Execute
        # Patching Event.wait with our custom function
        with patch.object(Event, "wait", new=custom_wait):
            thread, f_stop = start_update_client_manager_thread(driver, client_manager)
            # Wait until all nodes are registered via `client_manager.sample()`
            client_manager.sample(len(expected_node_ids))
            # Retrieve all nodes in `client_manager`
            node_ids = {proxy.node_id for proxy in client_manager.all().values()}
            # Update the GetNodesResponse and wait until the `client_manager` is updated
            driver.get_node_ids.return_value = updated_expected_node_ids
            time.sleep(0.1)
            # Retrieve all nodes in `client_manager`
            updated_node_ids = {
                proxy.node_id for proxy in client_manager.all().values()
            }
            # Stop the thread
            f_stop.set()

        # Assert
        assert node_ids == set(expected_node_ids)
        assert updated_node_ids == set(updated_expected_node_ids)

        # Exit
        thread.join()