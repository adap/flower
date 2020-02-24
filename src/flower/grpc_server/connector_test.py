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
"""Tests for Connector class."""
from threading import Thread

from flower.grpc_server.connector import Connector
from flower.proto.transport_pb2 import ClientRequest, ServerResponse


def test_run():
    """Test run method."""
    # Prepare
    connector = Connector()
    result_expected = ClientRequest()

    # As connector.run is blocking we will need to put the ClientRequest
    # in a background thread
    def worker():
        """Simulate processing loop."""
        # Wait until the ServerResponse is available and extract
        # although here we do nothing with the return value
        _ = connector.get_response()
        # Put ClientRequest into connector
        connector.put_request(ClientRequest())

    Thread(target=worker).start()

    # Execute
    result_actual = connector.run(ServerResponse())

    # Assert
    assert result_actual == result_expected


def test_put_get_request():
    """Test put/get request methods."""
    # Prepare
    connector = Connector()
    result_expected = ClientRequest()

    # Execute
    connector.put_request(result_expected)
    result_actual = connector.get_request()

    # Assert
    assert result_actual == result_expected


def test_put_get_response():
    """Test put/get response methods."""
    # Prepare
    connector = Connector()
    result_expected = ServerResponse()

    # Execute
    connector.put_response(result_expected)
    result_actual = connector.get_response()

    # Assert
    assert result_actual == result_expected
