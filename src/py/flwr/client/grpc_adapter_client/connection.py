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
"""Contextmanager for a GrpcAdapter channel to the Flower server."""


from collections.abc import Iterator
from contextlib import contextmanager
from logging import ERROR
from typing import Callable, Optional, Union

from cryptography.hazmat.primitives.asymmetric import ec

from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.client.grpc_rere_client.grpc_adapter import GrpcAdapter
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.common.message import Message
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.typing import Fab, Run


@contextmanager
def grpc_adapter(  # pylint: disable=R0913
    server_address: str,
    insecure: bool,
    retry_invoker: RetryInvoker,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: Optional[Union[bytes, str]] = None,
    authentication_keys: Optional[  # pylint: disable=unused-argument
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
) -> Iterator[
    tuple[
        Callable[[], Optional[Message]],
        Callable[[Message], None],
        Optional[Callable[[], Optional[int]]],
        Optional[Callable[[], None]],
        Optional[Callable[[int], Run]],
        Optional[Callable[[str], Fab]],
    ]
]:
    """Primitives for request/response-based interaction with a server via GrpcAdapter.

    Parameters
    ----------
    server_address : str
        The IPv6 address of the server with `http://` or `https://`.
        If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"http://[::]:8080"`.
    insecure : bool
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    retry_invoker: RetryInvoker
        `RetryInvoker` object that will try to reconnect the client to the server
        after gRPC errors. If None, the client will only try to
        reconnect once after a failure.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        Path of the root certificate. If provided, a secure
        connection using the certificates will be established to an SSL-enabled
        Flower server. Bytes won't work for the REST API.
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Client authentication is not supported for this transport type.

    Returns
    -------
    receive : Callable
    send : Callable
    create_node : Optional[Callable]
    delete_node : Optional[Callable]
    get_run : Optional[Callable]
    get_fab : Optional[Callable]
    """
    if authentication_keys is not None:
        log(ERROR, "Client authentication is not supported for this transport type.")
    with grpc_request_response(
        server_address=server_address,
        insecure=insecure,
        retry_invoker=retry_invoker,
        max_message_length=max_message_length,
        root_certificates=root_certificates,
        authentication_keys=None,  # Authentication is not supported
        adapter_cls=GrpcAdapter,
    ) as conn:
        yield conn
