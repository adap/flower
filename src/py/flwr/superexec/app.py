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
from typing import Optional

import grpc

from flwr.common import EventType, event, log
from flwr.common.address import parse_address
from flwr.common.config import parse_config_args
from flwr.common.constant import EXEC_API_DEFAULT_ADDRESS
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.object_ref import load_app, validate

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

    # Start SuperExec API
    superexec_server: grpc.Server = run_superexec_api_grpc(
        address=address,
        executor=_load_executor(args),
        certificates=certificates,
        config=parse_config_args(
            [args.executor_config] if args.executor_config else args.executor_config
        ),
    )

    grpc_servers = [superexec_server]

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPEREXEC_LEAVE,
        grpc_servers=grpc_servers,
        bckg_threads=None,
    )

    superexec_server.wait_for_termination()


def _parse_args_run_superexec() -> argparse.ArgumentParser:
    """Parse command line arguments for SuperExec."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperExec",
    )
    parser.add_argument(
        "--address",
        help="SuperExec (gRPC) server address (IPv4, IPv6, or a domain name)",
        default=EXEC_API_DEFAULT_ADDRESS,
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
        help="Key-value pairs for the executor config, separated by spaces. "
        'For example:\n\n`--executor-config \'superlink="superlink:9091" '
        'root-certificates="certificates/superlink-ca.crt"\'`',
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
    return parser


def _try_obtain_certificates(
    args: argparse.Namespace,
) -> Optional[tuple[bytes, bytes, bytes]]:
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
