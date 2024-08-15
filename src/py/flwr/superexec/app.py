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
"""Flower SuperExec app."""

import argparse
import sys
from logging import INFO, WARN
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple
import csv

import grpc
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

from flwr.common import EventType, event, log
from flwr.common.address import parse_address
from flwr.common.config import parse_config_args
from flwr.common.constant import SUPEREXEC_DEFAULT_ADDRESS
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.object_ref import load_app, validate
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.superexec.superexec_interceptor import SuperExecInterceptor

from .exec_grpc import run_superexec_api_grpc
from .executor import Executor


def run_superexec() -> None:
    """Run Flower SuperExec."""
    log(INFO, "Starting Flower SuperExec")

    event(EventType.RUN_SUPEREXEC_ENTER)

    args = _parse_args_run_superexec().parse_args()

    # Parse IP address
    parsed_address = parse_address(args.address)
    if not parsed_address:
        sys.exit(f"SuperExec IP address ({args.address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Obtain certificates
    certificates = _try_obtain_certificates(args)

    # Obtain keys for user authentication
    maybe_keys = _try_setup_user_authentication(args, certificates)
    interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None
    if maybe_keys is not None:
        (
            user_public_keys,
            superexec_private_key,
            superexec_public_key,
        ) = maybe_keys
        log(
            INFO,
            "User authentication enabled with %d known public keys",
            len(user_public_keys),
        )
        interceptors = [SuperExecInterceptor(user_public_keys, private_key_to_bytes(superexec_private_key), public_key_to_bytes(superexec_public_key))]
    
    # Start SuperExec API
    superexec_server: grpc.Server = run_superexec_api_grpc(
        address=address,
        executor=_load_executor(args),
        certificates=certificates,
        config=parse_config_args([args.executor_config]),
        interceptors=interceptors,
    )

    grpc_servers = [superexec_server]

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPEREXEC_LEAVE,
        grpc_servers=grpc_servers,
        bckg_threads=None,
    )

    superexec_server.wait_for_termination()


def _try_setup_user_authentication(
    args: argparse.Namespace,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
) -> Optional[Tuple[Set[bytes], ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]]:
    if (
        not args.auth_list_public_keys
        and not args.auth_superexec_private_key
        and not args.auth_superexec_public_key
    ):
        return None

    if (
        not args.auth_list_public_keys
        or not args.auth_superexec_private_key
        or not args.auth_superexec_public_key
    ):
        sys.exit(
            "Authentication requires providing file paths for "
            "'--auth-list-public-keys', '--auth-superexec-private-key' and "
            "'--auth-superexec-public-key'. Provide all three to enable authentication."
        )

    if certificates is None:
        sys.exit(
            "Authentication requires secure connections. "
            "Please provide certificate paths to `--ssl-certfile`, "
            "`--ssl-keyfile`, and `—-ssl-ca-certfile` and try again."
        )

    client_keys_file_path = Path(args.auth_list_public_keys)
    if not client_keys_file_path.exists():
        sys.exit(
            "The provided path to the known public keys CSV file does not exist: "
            f"{client_keys_file_path}. "
            "Please provide the CSV file path containing known public keys "
            "to '--auth-list-public-keys'."
        )

    client_public_keys: Set[bytes] = set()

    try:
        ssh_private_key = load_ssh_private_key(
            Path(args.auth_superexec_private_key).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the private key file in "
            "'--auth-superexec-private-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again."
        )

    try:
        ssh_public_key = load_ssh_public_key(
            Path(args.auth_superexec_public_key).read_bytes()
        )
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the public key file in "
            "'--auth-superexec-public-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again."
        )

    with open(client_keys_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for element in row:
                public_key = load_ssh_public_key(element.encode())
                if isinstance(public_key, ec.EllipticCurvePublicKey):
                    client_public_keys.add(public_key_to_bytes(public_key))
                else:
                    sys.exit(
                        "Error: Unable to parse the public keys in the CSV "
                        "file. Please ensure that the CSV file path points to a valid "
                        "known SSH public keys files and try again."
                    )
        return (
            client_public_keys,
            ssh_private_key,
            ssh_public_key,
        )


def _parse_args_run_superexec() -> argparse.ArgumentParser:
    """Parse command line arguments for SuperExec."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperExec",
    )
    parser.add_argument(
        "--address",
        help="SuperExec (gRPC) server address (IPv4, IPv6, or a domain name)",
        default=SUPEREXEC_DEFAULT_ADDRESS,
    )
    parser.add_argument(
        "--executor",
        help="For example: `deployment:exec` or `project.package.module:wrapper.exec`.",
        default="flwr.superexec.deployment:executor",
    )
    parser.add_argument(
        "--executor-dir",
        help="The directory for the executor.",
        default=".",
    )
    parser.add_argument(
        "--executor-config",
        help="Key-value pairs for the executor config, separated by commas. "
        'For example:\n\n`--executor-config superlink="superlink:9091",'
        'root-certificates="certificates/superlink-ca.crt"`',
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the SuperExec without HTTPS, regardless of whether certificate "
        "paths are provided. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="SuperExec server SSL certificate file (as a path str) "
        "to create a secure connection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="SuperExec server SSL private key file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--ssl-ca-certfile",
        help="SuperExec server SSL CA certificate file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--auth-list-public-keys",
        type=str,
        help="A CSV file (as a path str) containing a list of known public "
        "keys to enable user authentication.",
    )
    parser.add_argument(
        "--auth-superexec-private-key",
        type=str,
        help="The SuperExec's private key (as a path str) to enable user authentication.",
    )
    parser.add_argument(
        "--auth-superexec-public-key",
        type=str,
        help="The SuperExec's public key (as a path str) to enable user authentication.",
    )
    return parser


def _try_obtain_certificates(
    args: argparse.Namespace,
) -> Optional[Tuple[bytes, bytes, bytes]]:
    # Obtain certificates
    if args.insecure:
        log(WARN, "Option `--insecure` was set. Starting insecure HTTP server.")
        return None
    # Check if certificates are provided
    if args.ssl_certfile and args.ssl_keyfile and args.ssl_ca_certfile:
        if not Path(args.ssl_ca_certfile).is_file():
            sys.exit("Path argument `--ssl-ca-certfile` does not point to a file.")
        if not Path(args.ssl_certfile).is_file():
            sys.exit("Path argument `--ssl-certfile` does not point to a file.")
        if not Path(args.ssl_keyfile).is_file():
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
            "connection in SuperExec server (gRPC-rere)."
        )
    sys.exit(
        "Certificates are required unless running in insecure mode. "
        "Please provide certificate paths to `--ssl-certfile`, "
        "`--ssl-keyfile`, and `—-ssl-ca-certfile` or run the server "
        "in insecure mode using '--insecure' if you understand the risks."
    )


def _load_executor(
    args: argparse.Namespace,
) -> Executor:
    """Get the executor plugin."""
    executor_ref: str = args.executor
    valid, error_msg = validate(executor_ref, project_dir=args.executor_dir)
    if not valid and error_msg:
        raise LoadExecutorError(error_msg) from None

    executor = load_app(executor_ref, LoadExecutorError, args.executor_dir)

    if not isinstance(executor, Executor):
        raise LoadExecutorError(
            f"Attribute {executor_ref} is not of type {Executor}",
        ) from None

    return executor


class LoadExecutorError(Exception):
    """Error when trying to load `Executor`."""
