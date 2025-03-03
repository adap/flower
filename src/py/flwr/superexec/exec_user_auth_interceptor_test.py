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
"""Flower Exec API auth interceptor tests."""


import unittest
from unittest.mock import MagicMock

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

from flwr.common.typing import UserInfo
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetLoginDetailsRequest,
    ListRunsRequest,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
)
from flwr.superexec.exec_user_auth_interceptor import (
    ExecUserAuthInterceptor,
    shared_user_info,
)


class DummyUnaryUnaryHandler:
    """Dummy handler for unary-unary RPCs."""

    unary_unary = staticmethod(lambda request, context: "dummy_response")
    unary_stream = None
    request_deserializer = None
    response_serializer = None


class DummyUnsupportedHandler:
    """Dummy handler for unsupported RPC types."""

    unary_unary = None
    unary_stream = None
    request_deserializer = None
    response_serializer = None


class TestExecUserAuthInterceptor(unittest.TestCase):
    """Test the ExecUserAuthInterceptor authentication logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Set a known default for shared_user_info and store the token.
        self.default_user_info = UserInfo(user_id=None, user_name=None)
        self.token = shared_user_info.set(self.default_user_info)
        self.expected_user_info = UserInfo(user_id="user_id", user_name="user_name")

    def tearDown(self) -> None:
        """Reset shared_user_info to its previous state."""
        shared_user_info.reset(self.token)

    # pylint: disable=unused-argument
    def dummy_unary_unary_continuation(
        self, handler_call_details: grpc.HandlerCallDetails
    ) -> DummyUnaryUnaryHandler:
        """."""
        return DummyUnaryUnaryHandler()

    @parameterized.expand(
        [
            (GetAuthTokensRequest()),
            (GetLoginDetailsRequest(),),
        ]
    )  # type: ignore
    def test_unary_unary_login_request_successful(self, request: GrpcMessage) -> None:
        """Test unary-unary RPC call for a login request.

        Occurs for requests that are GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_auth_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_auth_plugin.validate_tokens_in_metadata.return_value = (False, None)
        interceptor = ExecUserAuthInterceptor(auth_plugin=dummy_auth_plugin)
        intercepted_handler = interceptor.intercept_service(
            self.dummy_unary_unary_continuation, handler_call_details
        )

        # Execute
        response = intercepted_handler.unary_unary(dummy_request, dummy_context)

        # Assert response is as expected
        self.assertEqual(response, "dummy_response")
        # Assert `shared_user_info` is not set
        user_info_from_context = shared_user_info.get()
        self.assertIsNone(user_info_from_context.user_id)
        self.assertIsNone(user_info_from_context.user_name)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_unary_login_request_unsuccessful(self, request: GrpcMessage) -> None:
        """Test unary-unary RPC call not successful when authentication fails.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_auth_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_auth_plugin.validate_tokens_in_metadata.return_value = (False, None)
        dummy_auth_plugin.refresh_tokens.return_value = None
        interceptor = ExecUserAuthInterceptor(auth_plugin=dummy_auth_plugin)
        intercepted_handler = interceptor.intercept_service(
            self.dummy_unary_unary_continuation, handler_call_details
        )

        # Execute & Assert: interceptor should abort with UNAUTHENTICATED
        with self.assertRaises(grpc.RpcError):
            intercepted_handler.unary_unary(dummy_request, dummy_context)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_unary_validate_tokens_successful(self, request) -> None:
        """Test unary-unary RPC call when token is valid.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_auth_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating valid tokens
        dummy_auth_plugin.validate_tokens_in_metadata.return_value = (
            True,
            self.expected_user_info,
        )
        interceptor = ExecUserAuthInterceptor(auth_plugin=dummy_auth_plugin)
        intercepted_handler = interceptor.intercept_service(
            self.dummy_unary_unary_continuation, handler_call_details
        )

        # Execute
        response = intercepted_handler.unary_unary(dummy_request, dummy_context)

        # Assert response is as expected
        self.assertEqual(response, "dummy_response")
        # Assert `shared_user_info` is set
        user_info_from_context = shared_user_info.get()
        self.assertEqual(
            user_info_from_context.user_id, self.expected_user_info.user_id
        )
        self.assertEqual(
            user_info_from_context.user_name, self.expected_user_info.user_name
        )

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_unary_refresh_tokens_successful(self, request) -> None:
        """Test unary-unary RPC call when tokens are refreshed successfully.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_auth_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_auth_plugin.validate_tokens_in_metadata.return_value = (False, None)
        # Set up refresh tokens
        expected_refresh_tokens_value = [("new-token", "value")]
        dummy_auth_plugin.refresh_tokens.return_value = expected_refresh_tokens_value

        interceptor = ExecUserAuthInterceptor(auth_plugin=dummy_auth_plugin)
        intercepted_handler = interceptor.intercept_service(
            self.dummy_unary_unary_continuation, handler_call_details
        )

        # Execute
        response = intercepted_handler.unary_unary(dummy_request, dummy_context)

        # Assert response is as expected and initial metadata was sent
        self.assertEqual(response, "dummy_response")
        # Assert refresh tokens were sent in initial metadata
        dummy_context.send_initial_metadata.assert_called_once_with(
            expected_refresh_tokens_value
        )
