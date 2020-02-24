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
from typing import Tuple
from unittest.mock import MagicMock

from google.protobuf.json_format import MessageToDict

from flower.grpc_server.flower_service_servicer import FlowerServiceServicer
from flower.proto.transport_pb2 import ClientInfo, ClientRequest, ServerResponse

CLIENT_INFO = ClientInfo(gpu=True)
CLIENT_REQUEST_CONNECT = ClientRequest(connect=ClientRequest.Connect(info=CLIENT_INFO))
CLIENT_REQUEST_TRAIN = ClientRequest(weight_update=ClientRequest.WeightUpdate())
SERVER_RESPONSE = ServerResponse()
CLIENT_CID = "some_client_cid"


def setup_mocks() -> Tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Create mocks for tests."""
    # Mock for the gRPC context argument
    context_mock = MagicMock()
    context_mock.peer.return_value = CLIENT_CID

    # Create a NetworkClient mock which we will use to test if correct
    # methods where called and requests are getting passed to it
    network_client_mock = MagicMock()
    network_client_mock.cid = CLIENT_CID
    network_client_mock.connector.get_response.return_value = ServerResponse()

    client_factory_mock = MagicMock()
    client_factory_mock.return_value = network_client_mock

    client_manager_mock = MagicMock()

    return context_mock, network_client_mock, client_factory_mock, client_manager_mock


def test_join():
    """Test Join method of FlowerServiceServicer."""
    # Prepare
    (
        context_mock,
        network_client_mock,
        client_factory_mock,
        client_manager_mock,
    ) = setup_mocks()

    # Create a instance of FlowerServiceServicer
    servicer = FlowerServiceServicer(
        client_manager=client_manager_mock, client_factory=client_factory_mock
    )

    # Define requests to be processed by FlowerServiceServicer instance
    requests = [CLIENT_REQUEST_CONNECT, CLIENT_REQUEST_TRAIN, CLIENT_REQUEST_TRAIN]
    requests_iter = iter(requests)

    # Execute
    response_iterator = servicer.Join(requests_iter, context_mock)

    # Assert
    num_responses = 0

    for response in response_iterator:
        num_responses += 1
        assert isinstance(response, ServerResponse)

    assert len(requests) == num_responses
    assert network_client_mock.cid == CLIENT_CID

    # After the first request is processed the CLIENT_REQUEST_CONNECT
    # the ClientFactory should have been called
    client_factory_mock.assert_called_once_with(CLIENT_CID, MessageToDict(CLIENT_INFO))

    # Check if the client was registered with the client_manager
    client_manager_mock.register.assert_called_once_with(network_client_mock)
    # Check if the client was unregistered with the client_manager
    client_manager_mock.unregister.assert_called_once_with(network_client_mock)
