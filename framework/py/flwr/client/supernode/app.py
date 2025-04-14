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
"""Flower SuperNode."""


import argparse
from logging import DEBUG, INFO, WARN
from pathlib import Path
from typing import Optional

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

from flwr.common import EventType, event
from flwr.common.args import try_obtain_root_certificates
from flwr.common.config import parse_config_args
from flwr.common.constant import (
    CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log

from ..app import start_client_internal
from ..clientapp.utils import get_load_client_app_fn


def run_supernode() -> None:
    """Run Flower SuperNode."""
    args = _parse_args_run_supernode().parse_args()

    log(INFO, "Starting Flower SuperNode")

    event(EventType.RUN_SUPERNODE_ENTER)

    # Check if both `--flwr-dir` and `--isolation` were set
    if args.flwr_dir is not None and args.isolation is not None:
        log(
            WARN,
            "Both `--flwr-dir` and `--isolation` were specified. "
            "Ignoring `--flwr-dir`.",
        )

    root_certificates = try_obtain_root_certificates(args, args.superlink)
    load_fn = get_load_client_app_fn(
        default_app_ref="",
        app_path=None,
        flwr_dir=args.flwr_dir,
        multi_app=True,
    )
    authentication_keys = _try_setup_client_authentication(args)

    log(DEBUG, "Isolation mode: %s", args.isolation)

    # Register handlers for graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERNODE_LEAVE,
        exit_message="SuperNode terminated gracefully.",
    )

    start_client_internal(
        server_address=args.superlink,
        load_client_app_fn=load_fn,
        transport=args.transport,
        root_certificates=root_certificates,
        insecure=args.insecure,
        authentication_keys=authentication_keys,
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
        node_config=parse_config_args(
            [args.node_config] if args.node_config else args.node_config
        ),
        flwr_path=args.flwr_dir,
        isolation=args.isolation,
        clientappio_api_address=args.clientappio_api_address,
    )


def _parse_args_run_supernode() -> argparse.ArgumentParser:
    """Parse flower-supernode command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperNode",
    )
    _parse_args_common(parser)
    parser.add_argument(
        "--flwr-dir",
        default=None,
        help="""The path containing installed Flower Apps.
        The default directory is:

        - `$FLWR_HOME/` if `$FLWR_HOME` is defined
        - `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined
        - `$HOME/.flwr/` in all other cases
        """,
    )
    parser.add_argument(
        "--isolation",
        default=ISOLATION_MODE_SUBPROCESS,
        required=False,
        choices=[
            ISOLATION_MODE_SUBPROCESS,
            ISOLATION_MODE_PROCESS,
        ],
        help="Isolation mode when running a `ClientApp` (`subprocess` by default, "
        "possible values: `subprocess`, `process`). Use `subprocess` to configure "
        "SuperNode to run a `ClientApp` in a subprocess. Use `process` to indicate "
        "that a separate independent process gets created outside of SuperNode.",
    )
    parser.add_argument(
        "--clientappio-api-address",
        default=CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
        help="ClientAppIo API (gRPC) server address (IPv4, IPv6, or a domain name). "
        f"By default, it is set to {CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS}.",
    )

    return parser


def _parse_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the client without HTTPS. By default, the client runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    ex_group = parser.add_mutually_exclusive_group()
    ex_group.add_argument(
        "--grpc-rere",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_GRPC_RERE,
        default=TRANSPORT_TYPE_GRPC_RERE,
        help="Use grpc-rere as a transport layer for the client.",
    )
    ex_group.add_argument(
        "--grpc-adapter",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_GRPC_ADAPTER,
        help="Use grpc-adapter as a transport layer for the client.",
    )
    ex_group.add_argument(
        "--rest",
        action="store_const",
        dest="transport",
        const=TRANSPORT_TYPE_REST,
        help="Use REST as a transport layer for the client.",
    )
    parser.add_argument(
        "--root-certificates",
        metavar="ROOT_CERT",
        type=str,
        help="Specifies the path to the PEM-encoded root certificate file for "
        "establishing secure HTTPS connections.",
    )
    parser.add_argument(
        "--superlink",
        default=FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
        help="SuperLink Fleet API address (IPv4, IPv6, or a domain name). If using the "
        "REST (experimental) transport, ensure your address is in the form "
        "`http://...` or `https://...` when TLS is enabled.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="The maximum number of times the client will try to reconnect to the"
        "SuperLink before giving up in case of a connection error. By default,"
        "it is set to None, meaning there is no limit to the number of tries.",
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=None,
        help="The maximum duration before the client stops trying to"
        "connect to the SuperLink in case of connection error. By default, it"
        "is set to None, meaning there is no limit to the total time.",
    )
    parser.add_argument(
        "--auth-supernode-private-key",
        type=str,
        help="The SuperNode's private key (as a path str) to enable authentication.",
    )
    parser.add_argument(
        "--auth-supernode-public-key",
        type=str,
        help="The SuperNode's public key (as a path str) to enable authentication.",
    )
    parser.add_argument(
        "--node-config",
        type=str,
        help="A space separated list of key/value pairs (separated by `=`) to "
        "configure the SuperNode. "
        "E.g. --node-config 'key1=\"value1\" partition-id=0 num-partitions=100'",
    )


def _try_setup_client_authentication(
    args: argparse.Namespace,
) -> Optional[tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]]:
    if not args.auth_supernode_private_key and not args.auth_supernode_public_key:
        return None

    if not args.auth_supernode_private_key or not args.auth_supernode_public_key:
        flwr_exit(ExitCode.SUPERNODE_NODE_AUTH_KEYS_REQUIRED)

    try:
        ssh_private_key = load_ssh_private_key(
            Path(args.auth_supernode_private_key).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        flwr_exit(
            ExitCode.SUPERNODE_NODE_AUTH_KEYS_INVALID,
            "Unable to parse the private key file.",
        )

    try:
        ssh_public_key = load_ssh_public_key(
            Path(args.auth_supernode_public_key).read_bytes()
        )
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        flwr_exit(
            ExitCode.SUPERNODE_NODE_AUTH_KEYS_INVALID,
            "Unable to parse the public key file.",
        )

    return (
        ssh_private_key,
        ssh_public_key,
    )
