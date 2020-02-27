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
"""Tests for GRPCBridge class."""
from threading import Thread
from typing import List

from flower.grpc_server.grpc_bridge import GRPCBridge
from flower.proto.transport_pb2 import ClientMessage, ServerMessage


def start_worker(rounds: int, bridge: GRPCBridge, results: List) -> Thread:
    """Simulate processing loop with five calls."""

    def _worker():
        # Wait until the ServerMessage is available and extract
        # although here we do nothing with the return value
        for _ in range(rounds):
            client_message = bridge.request(ServerMessage())

            if client_message is None:
                break

            results.append(client_message)

    thread = Thread(target=_worker)
    thread.start()

    return thread


def test_workflow_successful():
    """Test full workflow."""
    # Prepare
    rounds = 5
    client_messages_received = []

    bridge = GRPCBridge()
    server_message_iterator = bridge.server_message_iterator()

    worker_thread = start_worker(rounds, bridge, client_messages_received)

    # Execute
    # Simluate remote client side
    for _ in range(rounds):
        # First read the server message
        next(server_message_iterator)

        # Set the next client message
        bridge.set_client_message(ClientMessage())

    # Assert
    for msg in client_messages_received:
        assert isinstance(msg, ClientMessage)

    # Teardown
    worker_thread.join()


def test_workflow_interruption():
    """Test interrupted workflow."""
    # Prepare
    rounds = 5
    client_messages_received = []

    bridge = GRPCBridge()
    server_message_iterator = bridge.server_message_iterator()

    worker_thread = start_worker(rounds, bridge, client_messages_received)

    # Execute
    for i in range(rounds):
        try:
            next(server_message_iterator)
        except StopIteration:
            print("StopIteration raised")
            break

        bridge.set_client_message(ClientMessage())

        # Close the bridge after the third client message is set.
        # This should interrupt consumption of the message
        if i == 2:
            bridge.close()

    # Assert
    for msg in client_messages_received:
        assert isinstance(msg, ClientMessage)

    assert len(client_messages_received) == 2

    # Wait for thread join before finishing the test
    worker_thread.join(timeout=1)
