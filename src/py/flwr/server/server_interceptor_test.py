# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower server interceptor tests."""

import unittest

import grpc
from .app import ADDRESS_FLEET_API_GRPC_RERE, _run_fleet_api_grpc_rere
from flwr.client.app import _init_connection
from flwr.common.constant import TRANSPORT_TYPE_GRPC_RERE
from .superlink.state.state_factory import StateFactory

from flwr.common import GRPC_MAX_MESSAGE_LENGTH


class TestServerInterceptor(unittest.TestCase):
    """Server interceptor tests."""

    def setUp(self):
        """Initialize mock stub and server interceptor."""
        self._state_factory = StateFactory(":flwr-in-memory-state:")
        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            ADDRESS_FLEET_API_GRPC_RERE, self._state_factory
        )
        self._connection, self._address = _init_connection(
            TRANSPORT_TYPE_GRPC_RERE, ADDRESS_FLEET_API_GRPC_RERE
        )
        with self._connection(
            self._address,
            True,
            GRPC_MAX_MESSAGE_LENGTH,
        ) as conn:
            self._receive, self._send, self._create_node, self._delete_node = conn

    def tearDown(self):
        """Clean up grpc server."""
        self._server.stop(None)

    def test_successful_create_node_with_metadata(self) -> None:
        """Test server interceptor for create node."""
        self._create_node()
