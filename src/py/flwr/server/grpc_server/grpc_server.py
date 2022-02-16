# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Implements utility function to create a gRPC server."""


import concurrent.futures
import sys
from logging import ERROR
from typing import Optional, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto import transport_pb2_grpc
from flwr.server.client_manager import ClientManager
from flwr.server.grpc_server import flower_service_servicer as fss

INVALID_CERTIFICATES_ERR_MSG = """
    When setting any of root_certificate, certificate, or private_key,
    all of them need to be set.
"""


def valid_certificates(certificates: Tuple[bytes, bytes, bytes]) -> bool:
    """Validate certificates tuple."""
    is_valid = (
        all(isinstance(certificate, bytes) for certificate in certificates)
        and len(certificates) == 3
    )

    if not is_valid:
        log(ERROR, INVALID_CERTIFICATES_ERR_MSG)

    return is_valid


def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager: ClientManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> grpc.Server:
    """Create gRPC server and return instance of grpc.Server.

    If used in a main function server.wait_for_termination(timeout=None)
    should be called as otherwise the server will immediately stop.

    **SSL**
    To enable SSL you have to pass all of root_certificate, certificate,
    and private_key. Setting only some will make the process exit with code 1.

    Parameters
    ----------
    client_manager : ClientManager
        Instance of ClientManager
    server_address : str
        Server address in the form of HOST:PORT e.g. "[::]:8080"
    max_concurrent_workers : int
        Maximum number of clients the server can process before returning
        RESOURCE_EXHAUSTED status (default: 1000)
    max_message_length : int
        Maximum message length that the server can send or receive.
        Int valued in bytes. -1 means unlimited. (default: GRPC_MAX_MESSAGE_LENGTH)
    keepalive_time_ms : int
        Flower uses a default gRPC keepalive time of 210000ms (3 minutes 30 seconds)
        because some cloud providers (for example, Azure) agressively clean up idle
        TCP connections by terminating them after some time (4 minutes in the case
        of Azure). Flower does not use application-level keepalive signals and relies
        on the assumption that the transport layer will fail in cases where the
        connection is no longer active. `keepalive_time_ms` can be used to customize
        the keepalive interval for specific environments. The default Flower gRPC
        keepalive of 210000 ms (3 minutes 30 seconds) ensures that Flower can keep
        the long running streaming connection alive in most environments. The actual
        gRPC default of this setting is 7200000 (2 hours), which results in dropped
        connections in some cloud environments.

        These settings are related to the issue described here:
        - https://github.com/grpc/proposal/blob/master/A8-client-side-keepalive.md
        - https://github.com/grpc/grpc/blob/master/doc/keepalive.md
        - https://grpc.io/docs/guides/performance/

        Mobile Flower clients may choose to increase this value if their server
        environment allows long-running idle TCP connections.
        (default: 210000)
    certificates : Tuple[bytes, bytes, bytes] (default: None)
        Tuple containing root certificate, server certificate, and private key to
        start a secure SSL-enabled server. The tuple is expected to have three bytes
        elements in the following order:

            * CA certificate.
            * server certificate.
            * server private key.

    Returns
    -------
    server : grpc.Server
        An instance of a gRPC server which is already started

    Examples
    --------
    Starting a SSL-enabled server.

    >>> from pathlib import Path
    >>> start_grpc_server(
    >>>     client_manager=ClientManager(),
    >>>     server_address="localhost:8080",
    >>>     certificates=(
    >>>         Path("/crts/root.pem").read_bytes(),
    >>>         Path("/crts/localhost.crt").read_bytes(),
    >>>         Path("/crts/localhost.key").read_bytes(),
    >>>     ),
    >>> )
    """
    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    options = [
        # Maximum number of concurrent incoming streams to allow on a http2
        # connection. Int valued.
        ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
        # Maximum message length that the channel can send.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_send_message_length", max_message_length),
        # Maximum message length that the channel can receive.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_receive_message_length", max_message_length),
        # The gRPC default for this setting is 7200000 (2 hours). Flower uses a
        # customized default of 210000 (3 minutes and 30 seconds) to improve
        # compatibility with popular cloud providers. Mobile Flower clients may
        # choose to increase this value if their server environment allows
        # long-running idle TCP connections.
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        # Setting this to zero will allow sending unlimited keepalive pings in between
        # sending actual data frames.
        ("grpc.http2.max_pings_without_data", 0),
    ]

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=options,
    )

    servicer = fss.FlowerServiceServicer(client_manager)
    transport_pb2_grpc.add_FlowerServiceServicer_to_server(servicer, server)

    if certificates is not None:
        if not valid_certificates(certificates):
            sys.exit(1)

        root_certificate_b, certificate_b, private_key_b = certificates

        server_credentials = grpc.ssl_server_credentials(
            ((private_key_b, certificate_b),),
            root_certificates=root_certificate_b,
            # A boolean indicating whether or not to require clients to be
            # authenticated. May only be True if root_certificates is not None.
            # We are explicitly setting the current gRPC default to document
            # the option. For further reference see:
            # https://grpc.github.io/grpc/python/grpc.html#create-server-credentials
            require_client_auth=False,
        )
        server.add_secure_port(server_address, server_credentials)
    else:
        server.add_insecure_port(server_address)

    server.start()

    return server
