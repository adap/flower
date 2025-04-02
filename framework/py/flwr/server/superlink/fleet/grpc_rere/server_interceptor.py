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
"""Flower server interceptor."""


import datetime
from typing import Any, Callable, Optional, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.common.constant import (
    PUBLIC_KEY_HEADER,
    SIGNATURE_HEADER,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_HEADER,
    TIMESTAMP_TOLERANCE,
)
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_public_key,
    verify_signature,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
)
from flwr.server.superlink.linkstate import LinkStateFactory

MIN_TIMESTAMP_DIFF = -SYSTEM_TIME_TOLERANCE
MAX_TIMESTAMP_DIFF = TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE


def _unary_unary_rpc_terminator(
    message: str, code: Any = grpc.StatusCode.UNAUTHENTICATED
) -> grpc.RpcMethodHandler:
    def terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(code, message)
        raise RuntimeError("Should not reach this point")  # Make mypy happy

    return grpc.unary_unary_rpc_method_handler(terminate)


class AuthenticateServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Server interceptor for node authentication.

    Parameters
    ----------
    state_factory : LinkStateFactory
        A factory for creating new instances of LinkState.
    auto_auth : bool (default: False)
        If True, nodes are authenticated without requiring their public keys to be
        pre-stored in the LinkState. If False, only nodes with pre-stored public keys
        can be authenticated.
    """

    def __init__(self, state_factory: LinkStateFactory, auto_auth: bool = False):
        self.state_factory = state_factory
        self.auto_auth = auto_auth

    def intercept_service(  # pylint: disable=too-many-return-statements
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor authentication logic.

        Intercept all unary calls from nodes and authenticate nodes by validating auth
        metadata sent by the node. Continue RPC call if node is authenticated, else,
        terminate RPC call by setting context to abort.
        """
        # Filter out non-Fleet service calls
        if not handler_call_details.method.startswith("/flwr.proto.Fleet/"):
            return _unary_unary_rpc_terminator(
                "This request should be sent to a different service.",
                grpc.StatusCode.FAILED_PRECONDITION,
            )

        state = self.state_factory.state()
        metadata_dict = dict(handler_call_details.invocation_metadata)

        # Retrieve info from the metadata
        try:
            node_pk_bytes = cast(bytes, metadata_dict[PUBLIC_KEY_HEADER])
            timestamp_iso = cast(str, metadata_dict[TIMESTAMP_HEADER])
            signature = cast(bytes, metadata_dict[SIGNATURE_HEADER])
        except KeyError:
            return _unary_unary_rpc_terminator("Missing authentication metadata")

        if not self.auto_auth:
            # Abort the RPC call if the node public key is not found
            if node_pk_bytes not in state.get_node_public_keys():
                return _unary_unary_rpc_terminator("Public key not recognized")

        # Verify the signature
        node_pk = bytes_to_public_key(node_pk_bytes)
        if not verify_signature(node_pk, timestamp_iso.encode("ascii"), signature):
            return _unary_unary_rpc_terminator("Invalid signature")

        # Verify the timestamp
        current = now()
        time_diff = current - datetime.datetime.fromisoformat(timestamp_iso)
        # Abort the RPC call if the timestamp is too old or in the future
        if not MIN_TIMESTAMP_DIFF < time_diff.total_seconds() < MAX_TIMESTAMP_DIFF:
            return _unary_unary_rpc_terminator("Invalid timestamp")

        # Continue the RPC call
        expected_node_id = state.get_node_id(node_pk_bytes)
        if not handler_call_details.method.endswith("CreateNode"):
            # All calls, except for `CreateNode`, must provide a public key that is
            # already mapped to a `node_id` (in `LinkState`)
            if expected_node_id is None:
                return _unary_unary_rpc_terminator("Invalid node ID")
        # One of the method handlers in
        # `flwr.server.superlink.fleet.grpc_rere.fleet_server.FleetServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._wrap_method_handler(
            method_handler, expected_node_id, node_pk_bytes
        )

    def _wrap_method_handler(
        self,
        method_handler: grpc.RpcMethodHandler,
        expected_node_id: Optional[int],
        node_public_key: bytes,
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            # Verify the node ID
            if not isinstance(request, CreateNodeRequest):
                try:
                    if request.node.node_id != expected_node_id:  # type: ignore
                        raise ValueError
                except (AttributeError, ValueError):
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid node ID")

            response: GrpcMessage = method_handler.unary_unary(request, context)

            # Set the public key after a successful CreateNode request
            if isinstance(response, CreateNodeResponse):
                state = self.state_factory.state()
                try:
                    state.set_node_public_key(response.node.node_id, node_public_key)
                except ValueError as e:
                    # Remove newly created node if setting the public key fails
                    state.delete_node(response.node.node_id)
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, str(e))

            return response

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
