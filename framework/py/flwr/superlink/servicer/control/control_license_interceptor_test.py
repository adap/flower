# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower Control API license interceptor tests."""


import unittest
from collections.abc import Callable
from unittest.mock import MagicMock

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

from flwr.common.dummy_grpc_handlers_test import (
    get_noop_unary_stream_handler,
    get_noop_unary_unary_handler,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetLoginDetailsRequest,
    ListRunsRequest,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
)
from flwr.supercore.license_plugin import LicensePlugin

from .control_license_interceptor import ControlLicenseInterceptor

HandlerContinuation = Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler]


class TestControlLicenseInterceptor(unittest.TestCase):
    """Test the ControlLicenseInterceptor license-check logic."""

    def setUp(self) -> None:
        """Initialize."""
        self.license_plugin = MagicMock(spec=LicensePlugin)

    @parameterized.expand(
        [
            (ListRunsRequest(),),
            (StartRunRequest(),),
            (StopRunRequest(),),
            (StreamLogsRequest(),),
            (GetLoginDetailsRequest(),),
            (GetAuthTokensRequest(),),
        ]
    )  # type: ignore
    def test_license_interceptor_successful(self, request: GrpcMessage) -> None:
        """Test all RPC calls are successful when check_license() is successful."""
        # Prepare
        self.license_plugin.check_license.return_value = True
        interceptor = ControlLicenseInterceptor(self.license_plugin)
        dummy_ctx = MagicMock()
        handler_call_details = MagicMock()

        # Pick the right continuation based on stream vs unary
        continuation: HandlerContinuation = get_noop_unary_unary_handler
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler

        handler = interceptor.intercept_service(continuation, handler_call_details)

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            # Stream: collect into a list
            result = list(handler.unary_stream(request, dummy_ctx))
            self.assertEqual(result, ["stream response 1", "stream response 2"])
        else:
            # Unary: single return
            result = handler.unary_unary(request, dummy_ctx)
            self.assertEqual(result, "dummy_response")

        #  license_plugin.check_license called once
        self.license_plugin.check_license.assert_called_once()
        #  context.abort never called
        dummy_ctx.abort.assert_not_called()

    @parameterized.expand(
        [
            (ListRunsRequest(),),
            (StartRunRequest(),),
            (StopRunRequest(),),
            (StreamLogsRequest(),),
            (GetLoginDetailsRequest(),),
            (GetAuthTokensRequest(),),
        ]
    )  # type: ignore
    def test_license_failure(self, request: GrpcMessage) -> None:
        """Test all RPC calls are unsuccessful when check_license() fails."""
        # Prepare
        self.license_plugin.check_license.return_value = False
        interceptor = ControlLicenseInterceptor(self.license_plugin)
        dummy_ctx = MagicMock()
        handler_call_details = MagicMock()

        continuation: HandlerContinuation = get_noop_unary_unary_handler
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler

        handler = interceptor.intercept_service(continuation, handler_call_details)

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            with self.assertRaises(grpc.RpcError):
                _ = list(handler.unary_stream(request, dummy_ctx))
        else:
            with self.assertRaises(grpc.RpcError):
                _ = handler.unary_unary(request, dummy_ctx)

        # The interceptor should have called context.abort(...) once
        dummy_ctx.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "❗️ License check failed. Please contact the SuperLink administrator.",
        )
