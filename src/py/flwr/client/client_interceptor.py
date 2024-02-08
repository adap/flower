# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower client interceptor."""

import grpc
import collections
from typing import Callable, Union, Sequence, Tuple
from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    DeleteNodeRequest,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from cryptography.hazmat.primitives.asymmetric import ec
from common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac, 
    bytes_to_public_key, 
    generate_shared_key,
    public_key_to_bytes,
)

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

Request = Union[CreateNodeRequest, DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest]

def _get_value_from_tuples(key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]) -> Union[str, bytes]:
    return next((value[::-1] for key, value in tuples if key == key_string), "")

class _ClientCallDetails(
        collections.namedtuple(
            '_ClientCallDetails',
            ('method', 'timeout', 'metadata', 'credentials')),
        grpc.ClientCallDetails):
    pass

class AuthenticateClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey):
        self.private_key = private_key
        self.public_key = public_key

    def intercept_unary_unary(self, continuation: Callable, client_call_details: grpc.ClientCallDetails, request: Request):
        """Flower client interceptor."""
        metadata = []
        postprocess = False
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)

        if isinstance(request, CreateNodeRequest):
            metadata.append((_PUBLIC_KEY_HEADER, public_key_to_bytes(self.public_key)))
            postprocess = True

        elif isinstance(request, (DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest)):
            metadata.append(("auth-token", compute_hmac(self.shared_secret, request)))
        else:
            pass

        client_call_details = _ClientCallDetails(
            client_call_details.method, client_call_details.timeout, metadata,
            client_call_details.credentials)
        
        response = continuation(client_call_details, request)
        if postprocess:
            server_public_key_bytes = _get_value_from_tuples(_PUBLIC_KEY_HEADER, response.trailing_metadata)
            self.server_public_key = bytes_to_public_key(server_public_key_bytes)
            self.shared_secret = generate_shared_key(self.private_key, self.server_public_key)
        return response
            
