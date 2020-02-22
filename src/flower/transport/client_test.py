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
"""Tests for module server"""

import concurrent.futures
from typing import Iterator
from unittest.mock import patch

import grpc

import flower_testing
from flower.client_manager import SimpleClientManager
from flower.proto.transport_pb2 import ClientRequest, ServerResponse, Weights
from flower.transport.client import insecure_grpc_connection
from flower.transport.grpc_server import start_insecure_grpc_server

EXPECTED_RECONNECT_SECONDS = 60
EXPECTED_NUM_TRAIN_MESSAGES = 10

SERVER_RESPONSE_RECONNECT = ServerResponse(
    reconnect=ServerResponse.Reconnect(seconds=EXPECTED_RECONNECT_SECONDS)
)
SERVER_RESPONSE_TRAIN = ServerResponse(
    train=ServerResponse.Train(weights=Weights(weights=[]), epochs=10)
)
CLIENT_REQUEST_CONNECT = ClientRequest(connect=ClientRequest.Connect(uuid="123"))
CLIENT_REQUEST_WEIGHT_UPDATES = ClientRequest(
    weight_update=ClientRequest.WeightUpdate(
        weights=Weights(weights=[]), num_examples=10
    )
)


def mock_join(  # type: ignore # pylint: disable=invalid-name
    _self, request_iterator: Iterator[ClientRequest], _context: grpc.ServicerContext
) -> Iterator[ServerResponse]:
    """Serve as mock for the Join method of class FlowerServiceServicer."""
    counter = 0
    for _ in request_iterator:
        if counter < EXPECTED_NUM_TRAIN_MESSAGES:
            counter += 1
            yield SERVER_RESPONSE_TRAIN
        else:
            yield SERVER_RESPONSE_RECONNECT


@patch("flower.transport.flower_service_servicer.FlowerServiceServicer.Join", mock_join)
def test_integration_connection():
    """Create a server and establish a connection to it.

    Purpose of this integration test is to simulate multiple clients
    with multiple roundtrips between server and client.
    """
    # Prepare
    port = flower_testing.network.unused_tcp_port()

    _, server = start_insecure_grpc_server(
        client_manager=SimpleClientManager(), port=port
    )

    # Execute
    # Multiple clients in parallel
    def run_client():
        reconnect_seconds = 0
        num_train_messages = 0

        with insecure_grpc_connection(port=port) as conn:
            receive, send = conn

            # Send connect message
            send(CLIENT_REQUEST_CONNECT)

            # Setup processing loop
            while True:
                # Block until server responds with a message
                instruction = receive()

                if instruction.HasField("train"):
                    num_train_messages += 1
                    send(CLIENT_REQUEST_WEIGHT_UPDATES)
                elif instruction.HasField("reconnect"):
                    reconnect_seconds = instruction.reconnect.seconds
                    break
                else:
                    raise Exception("This should never happen")

        return reconnect_seconds, num_train_messages

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_client) for _ in range(3)]
        concurrent.futures.wait(futures)
        for future in futures:
            results.append(future.result())

    # Assert
    for res in results:
        reconnect_seconds, num_train_messages = res
        assert reconnect_seconds == EXPECTED_RECONNECT_SECONDS
        assert num_train_messages == EXPECTED_NUM_TRAIN_MESSAGES

    # Teardown
    server.stop(1)
