# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for ClientApp runtime wiring."""


import unittest
from unittest.mock import Mock, patch

import grpc

from flwr.supercore.interceptors import AppIoTokenClientInterceptor

from .run_clientapp import run_clientapp


class TestRunClientApp(unittest.TestCase):
    """Tests for `run_clientapp`."""

    def test_run_clientapp_adds_token_client_interceptor(self) -> None:
        """`run_clientapp` should add token interceptor to gRPC channel creation."""
        mock_channel = Mock()
        mock_stub = Mock()
        mock_heartbeat_sender = Mock()
        mock_heartbeat_sender.is_running = False

        with (
            patch(
                "flwr.supernode.runtime.run_clientapp.create_channel",
                return_value=mock_channel,
            ) as mock_create_channel,
            patch(
                "flwr.supernode.runtime.run_clientapp.ClientAppIoStub",
                return_value=mock_stub,
            ),
            patch("flwr.supernode.runtime.run_clientapp.wrap_stub"),
            patch(
                "flwr.supernode.runtime.run_clientapp.make_simple_grpc_retry_invoker"
            ),
            patch("flwr.supernode.runtime.run_clientapp.register_signal_handlers"),
            patch("flwr.supernode.runtime.run_clientapp.event"),
            patch(
                "flwr.supernode.runtime.run_clientapp.make_app_heartbeat_fn_grpc",
                return_value=lambda: True,
            ),
            patch(
                "flwr.supernode.runtime.run_clientapp.HeartbeatSender",
                return_value=mock_heartbeat_sender,
            ),
            patch(
                "flwr.supernode.runtime.run_clientapp.pull_appinputs",
                side_effect=grpc.RpcError(),
            ),
            patch("flwr.supernode.runtime.run_clientapp.flwr_exit"),
        ):
            run_clientapp(clientappio_api_address="127.0.0.1:9094", token="test-token")

        kwargs = mock_create_channel.call_args.kwargs
        interceptors = kwargs["interceptors"]
        self.assertIsNotNone(interceptors)
        assert interceptors is not None
        self.assertEqual(len(interceptors), 1)
        self.assertIsInstance(interceptors[0], AppIoTokenClientInterceptor)
