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
"""Common classes and functions for gRPC interceptor tests."""


from collections.abc import Iterator
from typing import Any

import grpc
from google.protobuf.message import Message as GrpcMessage


class DummyUnaryUnaryHandler:
    """Dummy unary-unary handler for testing."""

    unary_unary = staticmethod(lambda request, context: "dummy_response")
    unary_stream = None
    request_deserializer = None
    response_serializer = None


class DummyUnaryStreamHandler:
    """Dummy unary-stream handler for testing."""

    unary_unary = None
    unary_stream = staticmethod(
        lambda request, context: iter(["stream response 1", "stream response 2"])
    )
    request_deserializer = None
    response_serializer = None


class DummyUnsupportedHandler:
    """Dummy handler for unsupported RPC types."""

    unary_unary = None
    unary_stream = None
    request_deserializer = None
    response_serializer = None


class DummyUnaryUnaryHandlerException:
    """Dummy handler for unary-unary RPC calls that raises an Exception."""

    unary_unary = staticmethod(
        lambda request, context: (_ for _ in ()).throw(Exception("Test error"))
    )
    unary_stream = None
    request_deserializer = None
    response_serializer = None


# pylint: disable=unused-argument
def get_dummy_unary_unary_handler(
    handler_call_details: grpc.HandlerCallDetails,
) -> DummyUnaryUnaryHandler:
    """."""
    return DummyUnaryUnaryHandler()


# pylint: disable=unused-argument
def get_dummy_unary_stream_handler(
    handler_call_details: grpc.HandlerCallDetails,
) -> DummyUnaryStreamHandler:
    """."""
    return DummyUnaryStreamHandler()


def dummy_unary_stream_exception(
    request: GrpcMessage, context: grpc.ServicerContext  # pylint: disable=W0613
) -> Iterator[Any]:
    """Raise an Exception upon iteration for unary-stream RPC call."""

    def generator() -> Iterator[Any]:
        raise Exception("Test stream error")  # pylint: disable=W0719
        yield  # This yield is never reached. pylint: disable=W0101

    return generator()


# Dummy handler for unary_stream that uses the above function.
class DummyUnaryStreamHandlerException:
    """Dummy handler for unary-stream RPC calls that raises an Exception."""

    unary_unary = None
    unary_stream = staticmethod(dummy_unary_stream_exception)
    request_deserializer = None
    response_serializer = None
