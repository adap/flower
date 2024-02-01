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
"""Flower server interceptor."""

import grpc
from common.secure_aggregation.crypto.symmetric_encryption import generate_shared_key
from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    CreateNodeResponse,
)
from flwr.server.fleet.message_handler import message_handler
from flwr.server.state import StateFactory, State

def _unary_unary_rpc_terminator():

    def terminate(_, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied!")

    return grpc.unary_unary_rpc_method_handler(terminate)

def _create_node_with_public_key(state: State, server_public_key: bytes):

    def send_public_key(request: CreateNodeRequest, context: grpc.ServicerContext) -> CreateNodeResponse:
        context.set_trailing_metadata(
            (
                ("public-key", server_public_key),
            )
        )
        return message_handler.create_node(request, state)

    return grpc.unary_unary_rpc_method_handler(send_public_key)

def _handle_authentication(public_key, private_key):
    generate_shared_key(public_key, private_key)


class AuthenticateClientInterceptor(grpc.ServerInterceptor):

    def __init__(self, state_factory: StateFactory):
        self._public_key_header = "public-key"
        self._auth_token_header = "auth-token"
        self._state_factory = state_factory
        self._terminator = _unary_unary_rpc_terminator()
        self._create_node_handler = _create_node_with_public_key()

    def intercept_service(self, continuation, handler_call_details: grpc.HandlerCallDetails):
        if (self._header, self._value) in handler_call_details.invocation_metadata:
            grpc.unary_unary_rpc_method_handler
            return continuation(handler_call_details)
        else:
            return self._terminator
