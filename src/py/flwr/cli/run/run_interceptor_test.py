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
"""Flower client interceptor tests."""


import base64
import threading
import unittest
from concurrent import futures
from logging import DEBUG, INFO, WARN
from typing import Optional, Sequence, Tuple, Union

import grpc

from flwr.cli.run.run_interceptor import (
    _AUTH_TOKEN_HEADER,
    _PUBLIC_KEY_HEADER,
    Request,
    RunInterceptor,
)
from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.record import RecordSet
from flwr.common.retry_invoker import RetryInvoker, exponential
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    public_key_to_bytes,
)
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub


class _MockServicer:
    """Mock Servicer for Flower run."""

    def __init__(self) -> None:
        """Initialize mock servicer."""
        self._lock = threading.Lock()
        self._received_client_metadata: Optional[
            Sequence[Tuple[str, Union[str, bytes]]]
        ] = None
        self.superexec_private_key, self.superexec_public_key = generate_key_pairs()
        self._received_message_bytes: bytes = b""

    def unary_unary(self, request: Request, context: grpc.ServicerContext) -> Union[
        StartRunResponse,
        StreamLogsResponse,
    ]:
        """Handle unary call."""
        with self._lock:
            self._received_client_metadata = context.invocation_metadata()
            self._received_message_bytes = request.SerializeToString(True)

            if isinstance(request, StartRunRequest):
                print("StartRun, ", context.invocation_metadata())
                return StartRunResponse()

            return StreamLogsResponse()

    def received_client_metadata(
        self,
    ) -> Optional[Sequence[Tuple[str, Union[str, bytes]]]]:
        """Return received client metadata."""
        with self._lock:
            return self._received_client_metadata

    def received_message_bytes(self) -> bytes:
        """Return received message bytes."""
        with self._lock:
            return self._received_message_bytes


def _add_generic_handler(servicer: _MockServicer, server: grpc.Server) -> None:
    rpc_method_handlers = {
        "StartRun": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=StartRunRequest.FromString,
            response_serializer=StartRunResponse.SerializeToString,
        ),
        "StreamLogs": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=StreamLogsRequest.FromString,
            response_serializer=StreamLogsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "flwr.proto.Exec", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class TestRunInterceptor(unittest.TestCase):
    """Test for run interceptor user authentication."""

    def setUp(self) -> None:
        """Initialize mock server and client."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=(("grpc.so_reuseport", int(False)),),
        )
        self._servicer = _MockServicer()
        _add_generic_handler(self._servicer, self._server)
        port = self._server.add_insecure_port("[::]:0")
        self._server.start()

        self._user_private_key, self._user_public_key = generate_key_pairs()
        self._address = f"localhost:{port}"

        def on_channel_state_change(channel_connectivity: str) -> None:
            """Log channel connectivity."""
            log(DEBUG, channel_connectivity)

        channel = create_channel(
            server_address=self._address,
            insecure=True,
            root_certificates=None,
            max_message_length=GRPC_MAX_MESSAGE_LENGTH,
            interceptors=RunInterceptor(
                self._user_private_key,
                self._user_public_key,
                self._servicer.superexec_public_key,
            ),
        )
        channel.subscribe(on_channel_state_change)
        self.stub = ExecStub(channel)

    def test_user_auth_start_run(self) -> None:
        """Test user authentication during start run."""
        # Prepare
        req = StartRunRequest(
            fab_file=b"",
            override_config=None,
            federation_config=None,
        )

        # Execute
        self.stub.StartRun(req)

        shared_secret = generate_shared_key(
            self._servicer.superexec_private_key, self._user_public_key
        )
        expected_hmac = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, self._servicer.received_message_bytes())
        )
        actual_public_key = _get_value_from_tuples(
            _PUBLIC_KEY_HEADER, self._servicer.received_client_metadata()
        )
        actual_hmac = _get_value_from_tuples(
            _AUTH_TOKEN_HEADER, self._servicer.received_client_metadata()
        )
        expected_public_key = base64.urlsafe_b64encode(
            public_key_to_bytes(self._user_public_key)
        )

        # Assert
        assert actual_public_key == expected_public_key
        assert actual_hmac == expected_hmac


if __name__ == "__main__":
    unittest.main(verbosity=2)
