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
"""Flower client interceptor."""


import base64
import collections
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_public_key,
    compute_hmac,
    generate_shared_key,
    public_key_to_bytes,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PullTaskInsRequest,
    PushTaskResRequest,
)

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"

Request = Union[
    CreateNodeRequest, DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest
]


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails", ("method", "timeout", "metadata", "credentials")
    ),
    grpc.ClientCallDetails,  # type: ignore
):
    pass


class AuthenticateClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Client interceptor for client authentication."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        public_key: ec.EllipticCurvePublicKey,
    ):
        self.private_key = private_key
        self.public_key = public_key
        self.shared_secret = b""
        self.server_public_key: Optional[ec.EllipticCurvePublicKey] = None

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Flower client interceptor.

        Intercept unary call from client and add necessary authentication header in the
        RPC metadata.
        """
        metadata = []
        postprocess = False
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        if isinstance(request, CreateNodeRequest):
            metadata.append(
                (
                    _PUBLIC_KEY_HEADER,
                    base64.urlsafe_b64encode(public_key_to_bytes(self.public_key)),
                )
            )
            postprocess = True

        elif isinstance(
            request, (DeleteNodeRequest, PullTaskInsRequest, PushTaskResRequest)
        ):
            metadata.append(
                (
                    _PUBLIC_KEY_HEADER,
                    base64.urlsafe_b64encode(public_key_to_bytes(self.public_key)),
                )
            )
            metadata.append(
                (
                    _AUTH_TOKEN_HEADER,
                    base64.urlsafe_b64encode(
                        compute_hmac(
                            self.shared_secret, request.SerializeToString(True)
                        )
                    ),
                )
            )

        client_call_details = _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
        )

        response = continuation(client_call_details, request)
        if postprocess:
            server_public_key_bytes = base64.urlsafe_b64decode(
                _get_value_from_tuples(_PUBLIC_KEY_HEADER, response.initial_metadata())
            )
            self.server_public_key = bytes_to_public_key(server_public_key_bytes)
            self.shared_secret = generate_shared_key(
                self.private_key, self.server_public_key
            )
        return response
