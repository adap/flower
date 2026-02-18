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
"""Client interceptor for SuperExec signed authentication metadata."""


from collections.abc import Callable
from typing import Any

import grpc
from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.common.constant import (
    SUPEREXEC_PUBLIC_KEY_HEADER,
    SUPEREXEC_SIGNATURE_HEADER,
    SUPEREXEC_TIMESTAMP_HEADER,
)
from flwr.supercore.primitives.asymmetric import public_key_to_bytes, sign_message


class SuperExecAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach signed SuperExec authentication metadata to unary RPC calls."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        public_key: ec.EllipticCurvePublicKey,
    ):
        self.private_key = private_key
        self.public_key_bytes = public_key_to_bytes(public_key)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Attach signed auth metadata to a unary call."""
        metadata = list(client_call_details.metadata or [])

        timestamp = now().isoformat()
        method = client_call_details.method
        payload = f"{timestamp}\n{method}".encode()
        signature = sign_message(self.private_key, payload)

        metadata.append((SUPEREXEC_PUBLIC_KEY_HEADER, self.public_key_bytes))
        metadata.append((SUPEREXEC_TIMESTAMP_HEADER, timestamp))
        metadata.append((SUPEREXEC_SIGNATURE_HEADER, signature))

        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)
