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
"""Tests for ClientProxy class."""
from threading import Thread

from flower.grpc_server.client_proxy import ClientProxy
from flower.proto.transport_pb2 import ClientMessage, ServerMessage


def test_run():
    """Test run method."""
    # Prepare
    proxy = ClientProxy()
    client_message_expected = ClientMessage()

    # As connector.run is blocking we will need to put the ClientMessage
    # in a background thread
    def worker():
        """Simulate processing loop."""
        # Wait until the ServerMessage is available and extract
        # although here we do nothing with the return value
        _ = proxy.set_client_message_get_server_message(client_message=ClientMessage())

    Thread(target=worker).start()

    # Execute
    client_message_actual = proxy.set_server_message_get_client_message(
        server_message=ServerMessage()
    )

    # Assert
    assert client_message_actual == client_message_expected
