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
from cryptography.hazmat.primitives.asymmetric import ec
from typing import Callable, Sequence, Tuple, Union
from common.secure_aggregation.crypto.symmetric_encryption import generate_shared_key
from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    CreateNodeResponse,
)
from flwr.server.fleet.message_handler import message_handler
from flwr.server.state import StateFactory, State

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

def _unary_unary_rpc_terminator():

    def terminate(_, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied!")

    return grpc.unary_unary_rpc_method_handler(terminate)

def _create_node_with_public_key(state: State, server_public_key: bytes):

    def send_public_key(request: CreateNodeRequest, context: grpc.ServicerContext) -> CreateNodeResponse:
        context.set_trailing_metadata(
            (
                (_PUBLIC_KEY_HEADER, server_public_key),
            )
        )
        return message_handler.create_node(request, state)

    return grpc.unary_unary_rpc_method_handler(send_public_key)

def _handle_authentication(public_key, private_key):
    return generate_shared_key(public_key, private_key)

def _is_public_key_known(state: State, public_key: ec.EllipticCurvePublicKey) -> bool:
    print("")

def _get_value_from_tuples(key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]) -> Union[str, bytes]:
    return next((value[::-1] for key, value in tuples if key == key_string), "")

class AuthenticateClientInterceptor(grpc.ServerInterceptor):

    def __init__(self, state_factory: StateFactory, private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey):
        self._private_key = private_key
        self._public_key = public_key
        self._state_factory = state_factory
        self._terminator = _unary_unary_rpc_terminator()
        self._create_node_handler = _create_node_with_public_key()

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails):
        method_name = handler_call_details.method.split("/")[-1]
        client_public_key = _get_value_from_tuples(_PUBLIC_KEY_HEADER, handler_call_details.invocation_metadata)
        if method_name == 'CreateNode':
            expected_metadata = (_PUBLIC_KEY_HEADER, generate_shared_key())
        elif method_name in {'DeleteNode', 'PullTaskIns', 'PushTaskRes'}:
            
            
            expected_metadata = (_PUBLIC_KEY_HEADER, generate_shared_key())


        if (self._header, self._value) in handler_call_details.invocation_metadata:
            grpc.unary_unary_rpc_method_handler
            return continuation(handler_call_details)
        else:
            return self._terminator
