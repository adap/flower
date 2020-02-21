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
from flower.client_manager import SimpleClientManager
from flower.proto.transport_pb2 import ClientRequest, ServerResponse
from flower.transport.flower_service_servicer import FlowerServiceServicer

CLIENT_REQUEST = ClientRequest()
SERVER_RESPONSE = ServerResponse()


def test_join():
    """Test Join method of FlowerServiceServicer."""
    # Prepare
    client_manager = SimpleClientManager()
    servicer = FlowerServiceServicer(client_manager=client_manager)

    requests = [CLIENT_REQUEST for _ in range(10)]
    request_iterator = iter(requests)

    # Execute
    response_iterator = servicer.Join(request_iterator, {})

    # Assert
    num_responses = 0

    for response in response_iterator:
        num_responses += 1
        assert isinstance(response, ServerResponse)

    assert len(requests) == num_responses
