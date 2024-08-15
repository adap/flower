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


import base64
import unittest
from typing import Optional
from unittest.mock import MagicMock

import grpc

from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.superexec.exec_grpc import run_superexec_api_grpc
from flwr.superexec.executor import Executor, RunTracker, UserConfig
from flwr.superexec.simulation import SimulationEngine
from flwr.superexec.superexec_interceptor import (
    _AUTH_TOKEN_HEADER,
    _PUBLIC_KEY_HEADER,
    SuperExecInterceptor,
)


class MockExecutor(Executor):
    def set_config(
        self,
        config: UserConfig,
    ) -> None:
        """Mock set_config."""

    def start_run(
        self,
        fab_file: bytes,
        override_config: UserConfig,
        federation_config: UserConfig,
    ) -> Optional[RunTracker]:
        """Mock start_run."""


class TestSuperExecInterceptor(unittest.TestCase):  # pylint: disable=R0902
    """SuperExec interceptor tests."""

    def setUp(self) -> None:
        """Initialize mock stub and superexec interceptor."""
        address = "localhost:9093"
        self._user_private_key, self._user_public_key = generate_key_pairs()
        self._superexec_private_key, self._superexec_public_key = generate_key_pairs()

        self._superexec_interceptor = SuperExecInterceptor(
            {public_key_to_bytes(self._user_public_key)},
            private_key_to_bytes(self._superexec_private_key),
            public_key_to_bytes(self._superexec_public_key),
        )
        executor = MagicMock()
        run = MagicMock()
        executor.start_run = lambda _, __, ___: run
        self._server: grpc.Server = run_superexec_api_grpc(
            address=address,
            executor=executor,
            certificates=None,
            config={"num-supernodes": 10},
            interceptors=[self._superexec_interceptor],
        )

        self._channel = grpc.insecure_channel(address)
        self._start_run = self._channel.unary_unary(
            "/flwr.proto.Exec/StartRun",
            request_serializer=StartRunRequest.SerializeToString,
            response_deserializer=StartRunResponse.FromString,
        )
        self._stream_logs = self._channel.unary_unary(
            "/flwr.proto.Exec/StreamLogs",
            request_serializer=StreamLogsRequest.SerializeToString,
            response_deserializer=StreamLogsResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def test_successful_start_run_with_metadata(self) -> None:
        """Test superexec interceptor for creating node."""
        # Prepare
        request = StartRunRequest(
            fab_file=b"", override_config=None, federation_config=None
        )
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(self._user_public_key)
        )
        shared_secret = generate_shared_key(
            self._user_private_key,
            self._superexec_public_key,
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )

        # Execute
        response, call = self._start_run.with_call(
            request=request,
            metadata=(
                (_PUBLIC_KEY_HEADER, public_key_bytes),
                (_AUTH_TOKEN_HEADER, hmac_value),
            ),
        )

        expected_metadata = (
            _PUBLIC_KEY_HEADER,
            base64.urlsafe_b64encode(
                public_key_to_bytes(self._superexec_public_key)
            ).decode(),
        )

        # Assert
        assert isinstance(response, StartRunResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_unsuccessful_start_run_with_metadata(self) -> None:
        """Test superexec interceptor for creating node unsuccessfully."""
        # Prepare
        _, user_public_key = generate_key_pairs()
        public_key_bytes = base64.urlsafe_b64encode(
            public_key_to_bytes(user_public_key)
        )
        request = StartRunRequest(
            fab_file=b"", override_config=None, federation_config=None
        )
        shared_secret = generate_shared_key(
            self._user_private_key,
            self._superexec_public_key,
        )
        hmac_value = base64.urlsafe_b64encode(
            compute_hmac(shared_secret, request.SerializeToString(True))
        )

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self._start_run.with_call(
                request=StartRunRequest(
                    fab_file=b"", override_config=None, federation_config=None
                ),
                metadata=(
                    (_PUBLIC_KEY_HEADER, public_key_bytes),
                    (_AUTH_TOKEN_HEADER, hmac_value),
                ),
            )
