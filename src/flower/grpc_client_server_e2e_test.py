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
"""Tests for FlowerServiceServicer."""
import unittest
from threading import Thread
from typing import List

import numpy as np

from flower.client_manager import SimpleClientManager
from flower.grpc_client.connection import insecure_grpc_connection
from flower.grpc_server.grpc_server import start_insecure_grpc_server
from flower.proto.transport_pb2 import ClientInfo, ClientMessage, Disconnect, Weights
from flower_testing.network import unused_tcp_port

ONES = np.ones((2, 2))

CLIENT_MESSAGE_INFO = ClientMessage(info=ClientInfo(gpu=False))
CLIENT_MESSAGE_WEIGHT_UPDATES = ClientMessage(
    weight_update=ClientMessage.WeightUpdate(
        weights=Weights(weights=[]), num_examples=10
    )
)
CLIENT_MESSAGE_DISCONNECT = ClientMessage(disconnect=Disconnect(reason="POWER_OFF"))


def create_worker(port):
    """Create client and connect to server."""
    # pylint: disable=duplicate-code
    with insecure_grpc_connection(port=port) as conn:
        receive, send = conn

        message_count = 0

        # Setup processing loop
        while True:
            # Block until server responds with a message
            server_message = receive()
            message_count += 1

            if message_count >= 10:
                send(CLIENT_MESSAGE_DISCONNECT)
                break
            elif server_message.HasField("info"):
                send(CLIENT_MESSAGE_INFO)
            elif server_message.HasField("train"):
                send(CLIENT_MESSAGE_WEIGHT_UPDATES)
            elif server_message.HasField("reconnect"):
                # In a non test setup you might want to store the
                # seconds in a variable and use it to restart the
                # worker
                # reconnect_seconds = message.reconnect.seconds
                break
            else:
                raise Exception("Unhandled message")
        
        return


def create_workers(num_workers: int, port: int) -> List[Thread]:
    """Create num_workers workers each in a different thread."""
    threads = []
    for i in range(num_workers):
        threads.append(
            Thread(name=f"test_worker_{i}", target=create_worker, kwargs={"port": port})
        )

    for worker_thread in threads:
        worker_thread.start()

    return threads


class ClientServerE2ETestCase(unittest.TestCase):
    """Tests various E2E scenarios with client/server."""

    def setUp(self):
        """Create preconditions for test."""
        self.port = unused_tcp_port()
        self.client_manager = SimpleClientManager()
        self.server = start_insecure_grpc_server(
            client_manager=self.client_manager, port=self.port
        )

    def tearDown(self):
        """Stop server."""
        self.server.stop(3)

    def test_single_worker(self):
        """Test single connected worker"""
        # Prepare
        threads = create_workers(num_workers=1, port=self.port)
        self.client_manager.wait_for_clients(1)

        network_clients = self.client_manager.sample(1)

        for client in network_clients:
            for i in range(9):
                try:
                    weights, num_examples = client.fit([ONES])
                except Exception as ex:
                    print(ex)

        for worker_thread in threads:
            worker_thread.join()

