# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Tests for Flower ClientManager."""

import threading
import time
from unittest.mock import MagicMock

from flower.client_manager import SimpleClientManager
from flower.grpc_server.grpc_proxy_client import GRPCProxyClient


def test_simple_client_manager_register():
    """Tests if the register method works correctly"""
    # Prepare
    cid = "1"
    bridge = MagicMock()
    client = GRPCProxyClient(cid=cid, info={}, bridge=bridge)
    client_manager = SimpleClientManager()

    # Execute
    first = client_manager.register(client)
    second = client_manager.register(client)

    # Assert
    assert first
    assert not second
    assert len(client_manager) == 1


def test_simple_client_manager_unregister():
    """Tests if the unregister method works correctly"""
    # Prepare
    cid = "1"
    bridge = MagicMock()
    client = GRPCProxyClient(cid=cid, info={}, bridge=bridge)
    client_manager = SimpleClientManager()
    client_manager.register(client)

    # Execute
    client_manager.unregister(client)

    # Assert
    assert len(client_manager) == 0


def test_wait_for_clients():
    """Test if wait for clients is blocking correctly."""
    # Prepare
    bridge = MagicMock()
    start_time = time.time()
    client_manager = SimpleClientManager()

    def add_clients():
        """Block for a second and register couple clients with client_manager."""
        time.sleep(1)

        # This usually takes less than 1ms so waiting for one second above
        # is sufficent in all reasonable scenarios although there is a
        # theoretical chance this test might fail in the assert section
        client_manager.register(GRPCProxyClient(cid="1", info={}, bridge=bridge))
        client_manager.register(GRPCProxyClient(cid="2", info={}, bridge=bridge))

    threading.Thread(target=add_clients).start()

    # Execute
    client_manager.wait_for_clients(2)

    # Assert
    elapsed_time = time.time() - start_time
    assert len(client_manager) == 2
    assert elapsed_time >= 1
