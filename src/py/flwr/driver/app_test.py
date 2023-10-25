# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Flower Driver app tests."""
# pylint: disable=no-self-use


import threading
import time
import unittest
from unittest.mock import Mock, patch

from flwr.driver.app import update_client_manager
from flwr.proto.driver_pb2 import CreateWorkloadResponse, GetNodesResponse
from flwr.proto.node_pb2 import Node
from flwr.server.client_manager import SimpleClientManager

from .driver import Driver


class TestClientManagerWithDriver(unittest.TestCase):
    """Tests for ClientManager.

    Considering multi-threading, all tests assume that the `update_client_manager()`
    updates the ClientManager every 3 seconds.
    """

    def test_simple_client_manager_update(self) -> None:
        """Tests if the node update works correctly."""
        # Prepare
        mock_grpc_driver = Mock()
        mock_grpc_driver.create_workload.return_value = CreateWorkloadResponse(
            workload_id=61016
        )
        expected_nodes = [Node(node_id=i, anonymous=False) for i in range(100)]
        expected_updated_nodes = [
            Node(node_id=i, anonymous=False) for i in range(80, 120)
        ]
        mock_grpc_driver.get_nodes.return_value = GetNodesResponse(nodes=expected_nodes)
        client_manager = SimpleClientManager()
        patcher = patch("flwr.driver.driver.GrpcDriver", return_value=mock_grpc_driver)
        patcher.start()
        driver = Driver()
        bool_ref = [True]

        # Execute
        thread = threading.Thread(
            target=update_client_manager,
            args=(
                driver,
                client_manager,
                bool_ref,
            ),
            daemon=True,
        )
        thread.start()
        # Wait until all nodes are registered via `client_manager.sample()`
        client_manager.sample(len(expected_nodes))
        # Retrieve all nodes in `client_manager`
        node_ids = {proxy.node_id for proxy in client_manager.all().values()}
        # Update the GetNodesResponse and wait until the `client_manager` is updated
        mock_grpc_driver.get_nodes.return_value = GetNodesResponse(
            nodes=expected_updated_nodes
        )
        while True:
            if len(client_manager) == len(expected_updated_nodes):
                break
            time.sleep(1.3)
        # Retrieve all nodes in `client_manager`
        updated_node_ids = {proxy.node_id for proxy in client_manager.all().values()}
        # Stop client manager update
        bool_ref[0] = False

        # Assert
        mock_grpc_driver.create_workload.assert_called_once()
        assert node_ids == {node.node_id for node in expected_nodes}
        assert updated_node_ids == {node.node_id for node in expected_updated_nodes}

        # Exit
        patcher.stop()
        thread.join()
