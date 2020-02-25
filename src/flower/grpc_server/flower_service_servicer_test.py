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
from unittest.mock import MagicMock

from google.protobuf.json_format import MessageToDict

from flower.client import NetworkClient
from flower.grpc_server.flower_service_servicer import (
    ClientInfoMessageError,
    ClientManagerRejectionError,
    FlowerServiceServicer,
    default_client_factory,
    is_client_message_info,
    is_not_client_message_info,
    register_client,
)
from flower.proto.transport_pb2 import ClientInfo, ClientMessage, ServerMessage

CLIENT_INFO = ClientInfo(gpu=True)
CLIENT_MESSAGE_INFO = ClientMessage(info=CLIENT_INFO)
CLIENT_MESSAGE_TRAIN = ClientMessage(weight_update=ClientMessage.WeightUpdate())
SERVER_MESSAGE = ServerMessage()
CLIENT_CID = "some_client_cid"


class FlowerServiceServicerTestCase(unittest.TestCase):
    """Test suite for class FlowerServiceServicer and helper functions."""

    def setUp(self) -> None:
        """Create mocks for tests."""
        # Mock for the gRPC context argument
        self.context_mock = MagicMock()
        self.context_mock.peer.return_value = CLIENT_CID

        # Create a NetworkClient mock which we will use to test if correct
        # methods where called and client_messages are getting passed to it
        self.network_client_mock = MagicMock()
        self.network_client_mock.cid = CLIENT_CID
        self.network_client_mock.proxy.set_client_message_get_server_message.return_value = (
            ServerMessage()
        )

        self.client_factory_mock = MagicMock()
        self.client_factory_mock.return_value = self.network_client_mock

        self.client_manager_mock = MagicMock()

    def test_default_client_factory(self):
        """Confirm that the default client factory returns a NetworkClient."""
        # Execute
        client = default_client_factory(cid="any", info={})

        # Assert
        self.assertIsInstance(client, NetworkClient)

    def test_register_client(self):
        """Test register_client function."""
        # Prepare
        self.client_manager_mock.register.return_value = True

        # Execute
        register_client(
            client_manager=self.client_manager_mock,
            client=self.network_client_mock,
            context=self.context_mock,
        )

        # call_args contains the arguments each wrapped in a unittest.mock.call object
        # which holds the args in wrapped in a tuple. Therefore we need to take [0][0]
        callback = self.context_mock.add_callback.call_args[0][0]
        callback()

        # Assert
        self.client_manager_mock.register.assert_called_once_with(
            self.network_client_mock
        )
        self.context_mock.add_callback.assert_called_once()
        self.client_manager_mock.unregister.assert_called_once_with(
            self.network_client_mock
        )

    def test_register_client_exception(self):
        """Test register_client function."""
        # Prepare
        self.client_manager_mock.register.return_value = False

        # Execute & Assert
        self.assertRaises(
            ClientManagerRejectionError,
            lambda: register_client(
                client_manager=self.client_manager_mock,
                client=self.network_client_mock,
                context=self.context_mock,
            ),
        )

    def test_is_connect_message_no_exception(self):
        """Test that no exception is thrown."""
        # pylint: disable=no-self-use
        # Prepare
        client_message = CLIENT_MESSAGE_INFO

        # Execute & Assert
        is_client_message_info(client_message)

    def test_is_connect_message_exception(self):
        """Test that no exception is thrown."""
        # Prepare
        client_message = CLIENT_MESSAGE_TRAIN

        # Execute & Assert
        self.assertRaises(
            ClientInfoMessageError, lambda: is_client_message_info(client_message)
        )

    def test_is_not_connect_message_no_exception(self):
        """Test that no exception is thrown."""
        # pylint: disable=no-self-use
        # Prepare
        client_message = CLIENT_MESSAGE_TRAIN

        # Execute & Assert
        is_not_client_message_info(client_message)

    def test_is_not_connect_message_exception(self):
        """Test that no exception is thrown."""
        # Prepare
        client_message = CLIENT_MESSAGE_INFO

        # Execute & Assert
        self.assertRaises(
            ClientInfoMessageError, lambda: is_not_client_message_info(client_message)
        )

    def test_join(self):
        """Test Join method of FlowerServiceServicer."""
        # Prepare

        # Create a instance of FlowerServiceServicer
        servicer = FlowerServiceServicer(
            client_manager=self.client_manager_mock,
            client_factory=self.client_factory_mock,
        )

        # Define client_messages to be processed by FlowerServiceServicer instance
        client_messages = [
            CLIENT_MESSAGE_INFO,
            CLIENT_MESSAGE_TRAIN,
            CLIENT_MESSAGE_TRAIN,
        ]
        client_messages_iter = iter(client_messages)

        # Execute
        server_message_iterator = servicer.Join(client_messages_iter, self.context_mock)

        # Assert
        num_server_messages = 0

        for server_message in server_message_iterator:
            num_server_messages += 1
            assert isinstance(server_message, ServerMessage)

        assert len(client_messages) == num_server_messages
        assert self.network_client_mock.cid == CLIENT_CID

        # After the first client_message is processed the CLIENT_MESSAGE_CONNECT
        # the ClientFactory should have been called
        self.client_factory_mock.assert_called_once_with(
            CLIENT_CID, MessageToDict(CLIENT_INFO)
        )

        # Check if the client was registered with the client_manager
        self.client_manager_mock.register.assert_called_once_with(
            self.network_client_mock
        )
        # Check if the client was unregistered with the client_manager
        self.client_manager_mock.unregister.assert_called_once_with(
            self.network_client_mock
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
