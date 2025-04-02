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
"""Flower Exec API event log interceptor tests."""


import unittest
from typing import Optional, Union
from unittest.mock import MagicMock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.dummy_grpc_handlers_test import (
    NoOpUnaryStreamHandlerException,
    NoOpUnaryUnaryHandlerException,
    NoOpUnsupportedHandler,
    get_noop_unary_stream_handler,
    get_noop_unary_unary_handler,
)
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.typing import Actor, Event, LogEntry, UserInfo
from flwr.superexec.exec_event_log_interceptor import ExecEventLogInterceptor
from flwr.superexec.exec_user_auth_interceptor import shared_user_info


class DummyLogPlugin(EventLogWriterPlugin):
    """Dummy log plugin for testing."""

    def __init__(self) -> None:
        self.logs: list[LogEntry] = []

    def compose_log_before_event(
        self,
        request: GrpcMessage,
        context: grpc.ServicerContext,
        user_info: Optional[UserInfo],
        method_name: str,
    ) -> LogEntry:
        """Compose pre-event log entry from the provided request and context."""
        return LogEntry(
            timestamp="before_timestamp",
            actor=Actor(
                actor_id=user_info.user_id if user_info else None,
                description=user_info.user_name if user_info else None,
                ip_address="1.2.3.4",
            ),
            event=Event(action=method_name, run_id=None, fab_hash=None),
            status="before",
        )

    def compose_log_after_event(  # pylint: disable=too-many-arguments,R0917
        self,
        request: GrpcMessage,
        context: grpc.ServicerContext,
        user_info: Optional[UserInfo],
        method_name: str,
        response: Optional[Union[GrpcMessage, BaseException]],
    ) -> LogEntry:
        """Compose post-event log entry from the provided response and context."""
        return LogEntry(
            timestamp="after_timestamp",
            actor=Actor(
                actor_id=user_info.user_id if user_info else None,
                description=user_info.user_name if user_info else None,
                ip_address="5.6.7.8",
            ),
            event=Event(action=method_name, run_id=None, fab_hash=None),
            status="after",
        )

    def write_log(self, log_entry: LogEntry) -> None:
        """Write the event log to the specified data sink."""
        self.logs.append(log_entry)


class TestExecEventLogInterceptor(unittest.TestCase):
    """Test the ExecEventLogInterceptor logging for different RPC call types."""

    def setUp(self) -> None:
        """Initialize."""
        self.log_plugin = DummyLogPlugin()
        self.interceptor = ExecEventLogInterceptor(log_plugin=self.log_plugin)
        # Because shared_user_info.get() is read-only, we need to set the user info
        # and store the token to reset it after the test.
        self.expected_user_info = UserInfo(user_id="user_id", user_name="user_name")
        self.token = shared_user_info.set(self.expected_user_info)

    def tearDown(self) -> None:
        """Cleanup."""
        # Reset shared_user_info to its previous state
        shared_user_info.reset(self.token)

    def get_expected_logs(self, method_name: str) -> list[LogEntry]:
        """Get the expected log entries for the given method name."""
        expected_logs = [
            LogEntry(
                timestamp="before_timestamp",
                actor=Actor(
                    actor_id=self.expected_user_info.user_id,
                    description=self.expected_user_info.user_name,
                    ip_address="1.2.3.4",
                ),
                event=Event(action=method_name, run_id=None, fab_hash=None),
                status="before",
            ),
            LogEntry(
                timestamp="after_timestamp",
                actor=Actor(
                    actor_id=self.expected_user_info.user_id,
                    description=self.expected_user_info.user_name,
                    ip_address="5.6.7.8",
                ),
                event=Event(action=method_name, run_id=None, fab_hash=None),
                status="after",
            ),
        ]
        return expected_logs

    def test_unary_unary_interceptor(self) -> None:
        """Test unary-unary RPC call logging."""
        handler_call_details = MagicMock()
        handler_call_details.method = "dummy_method"
        expected_method_name = handler_call_details.method
        continuation = get_noop_unary_unary_handler
        intercepted_handler = self.interceptor.intercept_service(
            continuation, handler_call_details
        )
        expected_logs = self.get_expected_logs(expected_method_name)

        # Execute: Invoke the intercepted unary_unary method
        dummy_request = MagicMock()
        dummy_context = MagicMock()
        response = intercepted_handler.unary_unary(dummy_request, dummy_context)

        # Assert: Verify response and that logs were written before and after.
        self.assertEqual(response, "dummy_response")
        self.assertEqual(self.log_plugin.logs, expected_logs)

    def test_unary_unary_interceptor_exception(self) -> None:
        """Test unary-unary RPC call when the handler raises a BaseException."""
        handler_call_details = MagicMock()
        handler_call_details.method = "exception_method"
        expected_method_name = handler_call_details.method

        # pylint: disable=unused-argument
        def continuation(
            handler_call_details: grpc.HandlerCallDetails,
        ) -> NoOpUnaryUnaryHandlerException:
            return NoOpUnaryUnaryHandlerException()

        intercepted_handler = self.interceptor.intercept_service(
            continuation, handler_call_details
        )
        dummy_request = MagicMock()
        dummy_context = MagicMock()

        # Execute & Assert
        # Invoking the unary_unary method raises a BaseException with the expected msg
        with self.assertRaises(BaseException) as context_manager:
            intercepted_handler.unary_unary(dummy_request, dummy_context)
        self.assertEqual(str(context_manager.exception), "Test error")

        # Assert that the expected logs should include the before log and the after
        # log (even though an exception occurred)
        expected_logs = self.get_expected_logs(expected_method_name)
        self.assertEqual(self.log_plugin.logs, expected_logs)

    def test_unary_stream_interceptor(self) -> None:
        """Test unary-stream RPC call logging."""
        handler_call_details = MagicMock()
        handler_call_details.method = "stream_method"
        expected_method_name = handler_call_details.method
        continuation = get_noop_unary_stream_handler
        intercepted_handler = self.interceptor.intercept_service(
            continuation, handler_call_details
        )
        expected_logs = self.get_expected_logs(expected_method_name)

        # Execute: Invoke the intercepted unary_stream method
        dummy_request = MagicMock()
        dummy_context = MagicMock()
        response_iterator = intercepted_handler.unary_stream(
            dummy_request, dummy_context
        )
        responses = list(response_iterator)

        # Assert: Verify the stream responses and that logs were written.
        self.assertEqual(responses, ["stream response 1", "stream response 2"])
        self.assertEqual(self.log_plugin.logs, expected_logs)

    def test_unary_stream_interceptor_exception(self) -> None:
        """Test unary-stream RPC call when the stream handler raises a BaseException."""
        handler_call_details = MagicMock()
        handler_call_details.method = "stream_exception_method"
        expected_method_name = handler_call_details.method

        # pylint: disable=unused-argument
        def continuation(
            handler_call_details: grpc.HandlerCallDetails,
        ) -> NoOpUnaryStreamHandlerException:
            return NoOpUnaryStreamHandlerException()

        intercepted_handler = self.interceptor.intercept_service(
            continuation, handler_call_details
        )
        dummy_request = MagicMock()
        dummy_context = MagicMock()

        # Execute & Assert
        # Ierating the response iterator raises the expected exception.
        with self.assertRaises(BaseException) as context_manager:
            list(intercepted_handler.unary_stream(dummy_request, dummy_context))
        self.assertEqual(str(context_manager.exception), "Test stream error")

        # Assert that the expected logs should include the before log and the after
        # log (even though an exception occurred)
        expected_logs = self.get_expected_logs(expected_method_name)
        self.assertEqual(self.log_plugin.logs, expected_logs)

    def test_unsupported_rpc_method(self) -> None:
        """Test that unsupported RPC method types raise NotImplementedError."""

        # pylint: disable=unused-argument
        def continuation(
            handler_call_details: grpc.HandlerCallDetails,
        ) -> NoOpUnsupportedHandler:
            return NoOpUnsupportedHandler()

        handler_call_details = MagicMock()
        with self.assertRaises(NotImplementedError):
            self.interceptor.intercept_service(continuation, handler_call_details)
