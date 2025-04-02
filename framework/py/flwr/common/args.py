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
"""Common Flower arguments."""


import argparse
import sys
from logging import DEBUG, ERROR, WARN
from os.path import isfile
from pathlib import Path
from typing import Optional, Union

from flwr.common.constant import TRANSPORT_TYPE_REST
from flwr.common.logger import log


def add_args_flwr_app_common(parser: argparse.ArgumentParser) -> None:
    """Add common Flower arguments for flwr-*app to the provided parser."""
    parser.add_argument(
        "--flwr-dir",
        default=None,
        help="""The path containing installed Flower Apps.
        By default, this value is equal to:

            - `$FLWR_HOME/` if `$FLWR_HOME` is defined
            - `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined
            - `$HOME/.flwr/` in all other cases
        """,
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server without HTTPS, regardless of whether certificate "
        "paths are provided. Data transmitted between the gRPC client and server "
        "is not encrypted. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )


def try_obtain_root_certificates(
    args: argparse.Namespace,
    grpc_server_address: str,
) -> Optional[Union[bytes, str]]:
    """Validate and return the root certificates."""
    root_cert_path: Optional[str] = args.root_certificates
    if args.insecure:
        if root_cert_path is not None:
            sys.exit(
                "Conflicting options: The '--insecure' flag disables HTTPS, "
                "but '--root-certificates' was also specified. Please remove "
                "the '--root-certificates' option when running in insecure mode, "
                "or omit '--insecure' to use HTTPS."
            )
        log(
            WARN,
            "Option `--insecure` was set. Starting insecure HTTP channel to %s.",
            grpc_server_address,
        )
        root_certificates = None
    else:
        # Load the certificates if provided, or load the system certificates
        if root_cert_path is None:
            log(
                WARN,
                "Both `--insecure` and `--root-certificates` were not set. "
                "Using system certificates.",
            )
            root_certificates = None
        elif not isfile(root_cert_path):
            log(ERROR, "Path argument `--root-certificates` does not point to a file.")
            sys.exit(1)
        else:
            root_certificates = Path(root_cert_path).read_bytes()
        log(
            DEBUG,
            "Starting secure HTTPS channel to %s "
            "with the following certificates: %s.",
            grpc_server_address,
            root_cert_path,
        )
    if args.transport == TRANSPORT_TYPE_REST:
        return root_cert_path
    return root_certificates


def try_obtain_server_certificates(
    args: argparse.Namespace,
) -> Optional[tuple[bytes, bytes, bytes]]:
    """Validate and return the CA cert, server cert, and server private key."""
    if args.insecure:
        log(
            WARN,
            "Option `--insecure` was set. Starting insecure HTTP server with "
            "unencrypted communication (TLS disabled). Proceed only if you understand "
            "the risks.",
        )
        return None
    # Check if certificates are provided
    if args.ssl_certfile and args.ssl_keyfile and args.ssl_ca_certfile:
        if not isfile(args.ssl_ca_certfile):
            sys.exit("Path argument `--ssl-ca-certfile` does not point to a file.")
        if not isfile(args.ssl_certfile):
            sys.exit("Path argument `--ssl-certfile` does not point to a file.")
        if not isfile(args.ssl_keyfile):
            sys.exit("Path argument `--ssl-keyfile` does not point to a file.")
        certificates = (
            Path(args.ssl_ca_certfile).read_bytes(),  # CA certificate
            Path(args.ssl_certfile).read_bytes(),  # server certificate
            Path(args.ssl_keyfile).read_bytes(),  # server private key
        )
        return certificates
    if args.ssl_certfile or args.ssl_keyfile or args.ssl_ca_certfile:
        sys.exit(
            "You need to provide valid file paths to `--ssl-certfile`, "
            "`--ssl-keyfile`, and `—-ssl-ca-certfile` to create a secure "
            "connection in Fleet API server (gRPC-rere)."
        )
    log(
        ERROR,
        "Certificates are required unless running in insecure mode. "
        "Please provide certificate paths to `--ssl-certfile`, "
        "`--ssl-keyfile`, and `—-ssl-ca-certfile` or run the server "
        "in insecure mode using '--insecure' if you understand the risks.",
    )
    sys.exit(1)
