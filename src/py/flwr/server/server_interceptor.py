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
from flwr.server.state.authentication import AuthenticationState
from flwr.common.secure_aggregation.crypto.symmetric_encryption import generate_shared_key, bytes_to_public_key, public_key_to_bytes, verify_hmac
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

def _is_public_key_known(state: AuthenticationState, public_key: bytes) -> bool:
    return public_key in state.get_client_public_keys()

def _get_value_from_tuples(key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]) -> Union[str, bytes]:
    return next((value[::-1] for key, value in tuples if key == key_string), "")

class AuthenticateServerInterceptor(grpc.ServerInterceptor):

    def __init__(self, state_factory: StateFactory, private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey):
        self._private_key = private_key
        self._public_key = public_key
        self._state_factory = state_factory
        self._terminator = _unary_unary_rpc_terminator()
        self._create_node_handler = _create_node_with_public_key()

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails):
        method_name = handler_call_details.method.split("/")[-1]
        client_public_key_bytes = _get_value_from_tuples(_PUBLIC_KEY_HEADER, handler_call_details.invocation_metadata)
        client_public_key = bytes_to_public_key(client_public_key_bytes)

        if _is_public_key_known(self._state_factory.state, client_public_key_bytes):
            if method_name == 'CreateNode':
                return _create_node_with_public_key(self._state_factory.state, self._public_key)
            elif method_name in {'DeleteNode', 'PullTaskIns', 'PushTaskRes'}:
                state: AuthenticationState = self._state_factory.state
                shared_secret = generate_shared_key(self._private_key, client_public_key)
                hmac = _get_value_from_tuples(_AUTH_TOKEN_HEADER, handler_call_details.invocation_metadata)
                if verify_hmac(shared_secret, )
                state.get_client_public_keys()
                expected_metadata = (_AUTH_TOKEN_HEADER, generate_shared_key())


        if (self._header, self._value) in handler_call_details.invocation_metadata:
            grpc.unary_unary_rpc_method_handler
            return continuation(handler_call_details)
        else:
            return self._terminator
        
    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails):
        client_public_key_bytes = _get_value_from_tuples(_PUBLIC_KEY_HEADER, handler_call_details.invocation_metadata)
        if _is_public_key_known(self._state_factory.state, client_public_key_bytes):
            message_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
            message_handler.
            return grpc.unary_unary_rpc_method_handler(message_handler.unary_unary, request_deserializer=message_handler.request_deserializer, response_serializer=message_handler.response_serializer)
            if message_handler is None:
                return
        else:
            return self._terminator

        handler_factory, next_handler_method = _get_factory_and_method(next_handler)
        

        def invoke_intercept_method(request_or_iterator, context):
            method_name = handler_call_details.method
            return self.intercept(
                next_handler_method,
                request_or_iterator,
                context,
                method_name,
            )

        return handler_factory(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer,
        )
