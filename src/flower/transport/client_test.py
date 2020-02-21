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

from flower.client_manager import SimpleClientManager
from flower.helper_test import unused_tcp_port
from flower.proto import transport_pb2_grpc
from flower.proto.transport_pb2 import ClientRequest, ServerResponse, Weights
from flower.transport import flower_service_servicer
from flower.transport.client import connection
from flower.transport.server import create_server

SERVER_RESPONSE_RECONNECT = ServerResponse(
    reconnect=ServerResponse.Reconnect(seconds=60)
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


def test_integration_connection(monkeypatch):
    """Create a server and establish a connection to it.

    Purpose of this integration test is to simulate multiple clients
    with multiple roundtrips between server and client.
    """
    # Prepare
    port = unused_tcp_port()
    expected_reconnect_seconds = 60
    expected_num_train_messages = 10

    class MockFlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
        """Mock for FlowerServiceServicer"""

        def __init__(self, client_manager):
            pass

        def Join(self, request_iterator, context):
            counter = 0
            for _ in request_iterator:
                if counter < expected_num_train_messages:
                    counter += 1
                    yield SERVER_RESPONSE_TRAIN
                else:
                    yield SERVER_RESPONSE_RECONNECT

    monkeypatch.setattr(
        flower_service_servicer, "FlowerServiceServicer", MockFlowerServiceServicer
    )

    _, server = create_server(client_manager=SimpleClientManager(), port=port)

    # Execute
    # Multiple clients in parallel
    def run_client():
        reconnect_seconds = 0
        num_train_messages = 0

        with connection(port=port) as conn:
            consume, dispatch = conn

            # Send connect message
            dispatch(CLIENT_REQUEST_CONNECT)

            # Setup processing loop
            while True:
                # Block until server responds with a message
                instruction = consume()

                if instruction.HasField("train"):
                    num_train_messages += 1
                    dispatch(CLIENT_REQUEST_WEIGHT_UPDATES)

                elif instruction.HasField("reconnect"):
                    reconnect_seconds = instruction.reconnect.seconds
                    break

                else:
                    # if message is empty
                    break

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
        assert reconnect_seconds == expected_reconnect_seconds
        assert num_train_messages == expected_num_train_messages

    # Teardown
    server.stop(1)
