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
"""Flower run interceptor."""


import base64
import collections
from typing import Any, Callable, Union

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_shared_key,
    public_key_to_bytes,
)
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StreamLogsRequest,
)
from flwr.common.auth_plugin import UserAuthPlugin

Request = Union[
    StartRunRequest,
    StreamLogsRequest,
]


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails", ("method", "timeout", "metadata", "credentials")
    ),
    grpc.ClientCallDetails,  # type: ignore
):
    """Details for each client call.
    The class will be passed on as the first argument in continuation function.
    In our case, `RunInterceptor` adds new metadata to the construct.
    """


class UserInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """User interceptor for user authentication."""

    def __init__(
        self,
        auth_plugin: UserAuthPlugin
    ):
        self.auth_plugin = auth_plugin

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Flower SuperExec Run interceptor.
        Intercept unary call from user and add necessary authentication header in the
        RPC metadata.
        """
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)

        self.auth_plugin.provide_auth_details(metadata)

        client_call_details = _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
        )

        response = continuation(client_call_details, request)

        return response
