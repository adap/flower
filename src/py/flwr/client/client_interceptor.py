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
from typing import Callable, Union
from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    DeleteNodeRequest,
    PullTaskInsRequest,
    PushTaskResRequest,
)

Request = Union[CreateNodeRequest, DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest]

class _ClientCallDetails(
        collections.namedtuple(
            '_ClientCallDetails',
            ('method', 'timeout', 'metadata', 'credentials')),
        grpc.ClientCallDetails):
    pass

class AuthenticateClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey):
        self.private_key = private_key
        
    def intercept_unary_unary(self, continuation: Callable, client_call_details: grpc.ClientCallDetails, request: Request):
        """Flower client interceptor."""
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)

        if isinstance(request, CreateNodeRequest):
            pass
        elif isinstance(request, (DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest)):
            pass
        else:
            pass

        metadata.append(())
        client_call_details = _ClientCallDetails(
            client_call_details.method, client_call_details.timeout, metadata,
            client_call_details.credentials)
        
        response = continuation(client_call_details, request)
        return response
            
