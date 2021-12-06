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
from typing import ByteString, Optional, Tuple, Union

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto import transport_pb2_grpc
from flwr.server.client_manager import ClientManager
from flwr.server.grpc_server import flower_service_servicer as fss

FILELIKE = Union[str, bytes]
SSLFILES = Tuple[FILELIKE, FILELIKE, FILELIKE]

INVALID_SSL_FILES_ERR_MSG = """
    When setting any of root_certificate, certificate, or private_key,
    you have to set all of them.
"""


def read_to_byte_string(file_like: FILELIKE) -> ByteString:
    """Read file_like and return as ByteString."""
    with open(file_like, "rb") as file:
        return file.read()


def valid_ssl_files(ssl_files: SSLFILES) -> bool:
    """Validate type SSLFILES and exit if invalid."""
    is_valid = all(ssl_files)

    if not is_valid:
        log(ERROR, INVALID_SSL_FILES_ERR_MSG)

    return is_valid


def start_grpc_server(
    client_manager: ClientManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    ssl_files: Optional[SSLFILES] = None,
) -> grpc.Server:
    """Create gRPC server and return instance of grpc.Server.

    If used in a main function server.wait_for_termination(timeout=None)
    should be called as otherwise the server will immediately stop.

    **SSL/TLS**
    To enable SSL/TLS you have to pass all of root_certificate, certificate,
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
    ssl_files : tuple of union of (str, bytes)
        Certificates to start secure SSL/TLS server. Expected parameter is a tuple
        with three elements in this order beeing a file like of

            * CA certificate.
            * server certificate.
            * server private key.

        (default: None)

    Returns
    -------
    server : grpc.Server
        An instance of a gRPC server which is already started

    Examples
    --------
    Starting a SSL/TLS enabled server.

    >>> start_grpc_server(
    >>>     client_manager=ClientManager(),
    >>>     server_address="localhost:8080",
    >>>     ssl_files=("/crts/root.pem", "/crts/localhost.crt", "/crts/localhost.key")
    >>> )
    """
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=[
            # Maximum number of concurrent incoming streams to allow on a http2
            # connection. Int valued.
            ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
            # Maximum message length that the channel can send.
            # Int valued, bytes. -1 means unlimited.
            ("grpc.max_send_message_length", max_message_length),
            # Maximum message length that the channel can receive.
            # Int valued, bytes. -1 means unlimited.
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )

    servicer = fss.FlowerServiceServicer(client_manager)
    transport_pb2_grpc.add_FlowerServiceServicer_to_server(servicer, server)

    if ssl_files is not None:
        if not valid_ssl_files(ssl_files):
            sys.exit(1)

        root_certificate_b, certificate_b, private_key_b = [
            read_to_byte_string(file_path) for file_path in ssl_files
        ]

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
