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
import inspect
import threading
import unittest
from collections.abc import Sequence
from concurrent import futures
from logging import DEBUG, INFO, WARN
from typing import Optional, Union, get_args

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, serde
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
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.fleet_pb2_grpc import FleetServicer
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns  # pylint: disable=E0611

from .client_interceptor import _AUTH_TOKEN_HEADER, _PUBLIC_KEY_HEADER, Request
from .connection import grpc_request_response


class _MockServicer:
    """Mock Servicer for Flower clients."""

    def __init__(self) -> None:
        """Initialize mock servicer."""
        self._lock = threading.Lock()
        self._received_client_metadata: Optional[
            Sequence[tuple[str, Union[str, bytes]]]
        ] = None
        self.server_private_key, self.server_public_key = generate_key_pairs()
        self._received_message_bytes: bytes = b""

    def unary_unary(
        self, request: Request, context: grpc.ServicerContext
    ) -> Union[
        CreateNodeResponse, DeleteNodeResponse, PushTaskResResponse, PullTaskInsResponse
    ]:
        """Handle unary call."""
        with self._lock:
            self._received_client_metadata = context.invocation_metadata()
            self._received_message_bytes = request.SerializeToString(deterministic=True)

            if isinstance(request, CreateNodeRequest):
                context.send_initial_metadata(
                    (
                        (
                            _PUBLIC_KEY_HEADER,
                            base64.urlsafe_b64encode(
                                public_key_to_bytes(self.server_public_key)
                            ),
                        ),
                    )
                )
                return CreateNodeResponse(node=Node(node_id=123))
            if isinstance(request, DeleteNodeRequest):
                return DeleteNodeResponse()
            if isinstance(request, PushTaskResRequest):
                return PushTaskResResponse()

            return PullTaskInsResponse(
                task_ins_list=[
                    TaskIns(
                        task=Task(
                            consumer=Node(node_id=123),
                            recordset=serde.recordset_to_proto(RecordSet()),
                        )
                    )
                ]
            )

    def received_client_metadata(
        self,
    ) -> Optional[Sequence[tuple[str, Union[str, bytes]]]]:
        """Return received client metadata."""
        with self._lock:
            return self._received_client_metadata

    def received_message_bytes(self) -> bytes:
        """Return received message bytes."""
        with self._lock:
            return self._received_message_bytes


def _add_generic_handler(servicer: _MockServicer, server: grpc.Server) -> None:
    rpc_method_handlers = {
        "CreateNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=CreateNodeRequest.FromString,
            response_serializer=CreateNodeResponse.SerializeToString,
        ),
        "DeleteNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=DeleteNodeRequest.FromString,
            response_serializer=DeleteNodeResponse.SerializeToString,
        ),
        "PullTaskIns": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PullTaskInsRequest.FromString,
            response_serializer=PullTaskInsResponse.SerializeToString,
        ),
        "PushTaskRes": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PushTaskResRequest.FromString,
            response_serializer=PushTaskResResponse.SerializeToString,
        ),
        "GetRun": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=GetRunRequest.FromString,
            response_serializer=GetRunResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "flwr.proto.Fleet", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


def _init_retry_invoker() -> RetryInvoker:
    return RetryInvoker(
        wait_gen_factory=exponential,
        recoverable_exceptions=grpc.RpcError,
        max_tries=1,
        max_time=None,
        on_giveup=lambda retry_state: (
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_success=lambda retry_state: (
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_backoff=lambda retry_state: (
            log(WARN, "Connection attempt failed, retrying...")
            if retry_state.tries == 1
            else log(
                DEBUG,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )
        ),
    )


class TestAuthenticateClientInterceptor(unittest.TestCase):
    """Test for client interceptor client authentication."""

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
        self._client_private_key, self._client_public_key = generate_key_pairs()

        self._connection = grpc_request_response
        self._address = f"localhost:{port}"

    def test_client_auth_create_node(self) -> None:
        """Test client authentication during create node."""
        # Prepare
        retry_invoker = _init_retry_invoker()

        # Execute
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            _, _, create_node, _, _, _ = conn
            assert create_node is not None
            create_node()

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            actual_public_key = _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, received_metadata
            )

            expected_public_key = base64.urlsafe_b64encode(
                public_key_to_bytes(self._client_public_key)
            )

            # Assert
            assert actual_public_key == expected_public_key

    def test_client_auth_delete_node(self) -> None:
        """Test client authentication during delete node."""
        # Prepare
        retry_invoker = _init_retry_invoker()

        # Execute
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            _, _, create_node, delete_node, _, _ = conn
            assert create_node is not None
            create_node()
            assert delete_node is not None
            delete_node()

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            shared_secret = generate_shared_key(
                self._servicer.server_private_key, self._client_public_key
            )
            expected_hmac = base64.urlsafe_b64encode(
                compute_hmac(shared_secret, self._servicer.received_message_bytes())
            )
            actual_public_key = _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, received_metadata
            )
            actual_hmac = _get_value_from_tuples(_AUTH_TOKEN_HEADER, received_metadata)
            expected_public_key = base64.urlsafe_b64encode(
                public_key_to_bytes(self._client_public_key)
            )

            # Assert
            assert actual_public_key == expected_public_key
            assert actual_hmac == expected_hmac

    def test_client_auth_receive(self) -> None:
        """Test client authentication during receive node."""
        # Prepare
        retry_invoker = _init_retry_invoker()

        # Execute
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            receive, _, create_node, _, _, _ = conn
            assert create_node is not None
            create_node()
            assert receive is not None
            receive()

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            shared_secret = generate_shared_key(
                self._servicer.server_private_key, self._client_public_key
            )
            expected_hmac = base64.urlsafe_b64encode(
                compute_hmac(shared_secret, self._servicer.received_message_bytes())
            )
            actual_public_key = _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, received_metadata
            )
            actual_hmac = _get_value_from_tuples(_AUTH_TOKEN_HEADER, received_metadata)
            expected_public_key = base64.urlsafe_b64encode(
                public_key_to_bytes(self._client_public_key)
            )

            # Assert
            assert actual_public_key == expected_public_key
            assert actual_hmac == expected_hmac

    def test_client_auth_send(self) -> None:
        """Test client authentication during send node."""
        # Prepare
        retry_invoker = _init_retry_invoker()
        message = Message(Metadata(0, "", 123, 0, "", "", 0, ""), RecordSet())

        # Execute
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            receive, send, create_node, _, _, _ = conn
            assert create_node is not None
            create_node()
            assert receive is not None
            receive()
            assert send is not None
            send(message)

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            shared_secret = generate_shared_key(
                self._servicer.server_private_key, self._client_public_key
            )
            expected_hmac = base64.urlsafe_b64encode(
                compute_hmac(shared_secret, self._servicer.received_message_bytes())
            )
            actual_public_key = _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, received_metadata
            )
            actual_hmac = _get_value_from_tuples(_AUTH_TOKEN_HEADER, received_metadata)
            expected_public_key = base64.urlsafe_b64encode(
                public_key_to_bytes(self._client_public_key)
            )

            # Assert
            assert actual_public_key == expected_public_key
            assert actual_hmac == expected_hmac

    def test_client_auth_get_run(self) -> None:
        """Test client authentication during send node."""
        # Prepare
        retry_invoker = _init_retry_invoker()

        # Execute
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            _, _, create_node, _, get_run, _ = conn
            assert create_node is not None
            create_node()
            assert get_run is not None
            get_run(0)

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            shared_secret = generate_shared_key(
                self._servicer.server_private_key, self._client_public_key
            )
            expected_hmac = base64.urlsafe_b64encode(
                compute_hmac(shared_secret, self._servicer.received_message_bytes())
            )
            actual_public_key = _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, received_metadata
            )
            actual_hmac = _get_value_from_tuples(_AUTH_TOKEN_HEADER, received_metadata)
            expected_public_key = base64.urlsafe_b64encode(
                public_key_to_bytes(self._client_public_key)
            )

            # Assert
            assert actual_public_key == expected_public_key
            assert actual_hmac == expected_hmac

    def test_without_servicer(self) -> None:
        """Test client authentication without servicer."""
        # Prepare
        self._server.stop(grace=None)
        retry_invoker = _init_retry_invoker()

        # Execute and Assert
        with self._connection(
            self._address,
            True,
            retry_invoker,
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            _, _, create_node, _, _, _ = conn
            assert create_node is not None
            create_node()

            assert self._servicer.received_client_metadata() is None

    def test_fleet_requests_included(self) -> None:
        """Test if all Fleet requests are included in the authentication mode."""
        # Prepare
        requests = get_args(Request)
        rpc_names = {req.__qualname__.removesuffix("Request") for req in requests}
        expected_rpc_names = {
            name
            for name, ref in inspect.getmembers(FleetServicer)
            if inspect.isfunction(ref)
        }

        # Assert
        assert expected_rpc_names == rpc_names


if __name__ == "__main__":
    unittest.main(verbosity=2)
