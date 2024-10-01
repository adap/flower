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
"""Flower SuperNode."""

import argparse
import sys
from logging import DEBUG, ERROR, INFO, WARN
from pathlib import Path
from typing import Optional

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)

from flwr.common import EventType, event
from flwr.common.config import parse_config_args
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
)
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log, warn_deprecated_feature

from ..app import (
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    start_client_internal,
)
from ..clientapp.utils import get_load_client_app_fn


def run_supernode() -> None:
    """Run Flower SuperNode."""
    log(INFO, "Starting Flower SuperNode")

    event(EventType.RUN_SUPERNODE_ENTER)

    args = _parse_args_run_supernode().parse_args()

    _warn_deprecated_server_arg(args)

    root_certificates = _get_certificates(args)
    load_fn = get_load_client_app_fn(
        default_app_ref="",
        app_path=args.app,
        flwr_dir=args.flwr_dir,
        multi_app=True,
    )
    authentication_keys = _try_setup_client_authentication(args)

    log(DEBUG, "Isolation mode: %s", args.isolation)

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
        isolation=args.isolation,
        supernode_address=args.supernode_address,
    )

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERNODE_LEAVE,
    )


def run_client_app() -> None:
    """Run Flower client app."""
    event(EventType.RUN_CLIENT_APP_ENTER)
    log(
        ERROR,
        "The command `flower-client-app` has been replaced by `flower-supernode`.",
    )
    log(INFO, "Execute `flower-supernode --help` to learn how to use it.")
    register_exit_handlers(event_type=EventType.RUN_CLIENT_APP_LEAVE)


def _warn_deprecated_server_arg(args: argparse.Namespace) -> None:
    """Warn about the deprecated argument `--server`."""
    if args.server != FLEET_API_GRPC_RERE_DEFAULT_ADDRESS:
        warn = "Passing flag --server is deprecated. Use --superlink instead."
        warn_deprecated_feature(warn)

        if args.superlink != FLEET_API_GRPC_RERE_DEFAULT_ADDRESS:
            # if `--superlink` also passed, then
            # warn user that this argument overrides what was passed with `--server`
            log(
                WARN,
                "Both `--server` and `--superlink` were passed. "
                "`--server` will be ignored. Connecting to the Superlink Fleet API "
                "at %s.",
                args.superlink,
            )
        else:
            args.superlink = args.server


def _get_certificates(args: argparse.Namespace) -> Optional[bytes]:
    """Load certificates if specified in args."""
    # Obtain certificates
    if args.insecure:
        if args.root_certificates is not None:
            sys.exit(
                "Conflicting options: The '--insecure' flag disables HTTPS, "
                "but '--root-certificates' was also specified. Please remove "
                "the '--root-certificates' option when running in insecure mode, "
                "or omit '--insecure' to use HTTPS."
            )
        log(
            WARN,
            "Option `--insecure` was set. "
            "Starting insecure HTTP client connected to %s.",
            args.superlink,
        )
        root_certificates = None
    else:
        # Load the certificates if provided, or load the system certificates
        cert_path = args.root_certificates
        if cert_path is None:
            root_certificates = None
        else:
            root_certificates = Path(cert_path).read_bytes()
        log(
            DEBUG,
            "Starting secure HTTPS client connected to %s "
            "with the following certificates: %s.",
            args.superlink,
            cert_path,
        )
    return root_certificates


def _parse_args_run_supernode() -> argparse.ArgumentParser:
    """Parse flower-supernode command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperNode",
    )

    parser.add_argument(
        "app",
        nargs="?",
        default=None,
        help="Specify the path of the Flower App to load and run the `ClientApp`. "
        "The `pyproject.toml` file must be located in the root of this path. "
        "When this argument is provided, the SuperNode will exclusively respond to "
        "messages from the corresponding `ServerApp` by matching the FAB ID and FAB "
        "version. An error will be raised if a message is received from any other "
        "`ServerApp`.",
    )
    _parse_args_common(parser)
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
        "--isolation",
        default=None,
        required=False,
        choices=[
            ISOLATION_MODE_SUBPROCESS,
            ISOLATION_MODE_PROCESS,
        ],
        help="Isolation mode when running `ClientApp` (optional, possible values: "
        "`subprocess`, `process`). By default, `ClientApp` runs in the same process "
        "that executes the SuperNode. Use `subprocess` to configure SuperNode to run "
        "`ClientApp` in a subprocess. Use `process` to indicate that a separate "
        "independent process gets created outside of SuperNode.",
    )
    parser.add_argument(
        "--supernode-address",
        default="0.0.0.0:9094",
        help="Set the SuperNode gRPC server address. Defaults to `0.0.0.0:9094`.",
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
        "--server",
        default=FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
        help="Server address",
    )
    parser.add_argument(
        "--superlink",
        default=FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
        help="SuperLink Fleet API (gRPC-rere) address (IPv4, IPv6, or a domain name)",
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
        sys.exit(
            "Authentication requires file paths to both "
            "'--auth-supernode-private-key' and '--auth-supernode-public-key'"
            "to be provided (providing only one of them is not sufficient)."
        )

    try:
        ssh_private_key = load_ssh_private_key(
            Path(args.auth_supernode_private_key).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the private key file in "
            "'--auth-supernode-private-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again."
        )

    try:
        ssh_public_key = load_ssh_public_key(
            Path(args.auth_supernode_public_key).read_bytes()
        )
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the public key file in "
            "'--auth-supernode-public-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again."
        )

    return (
        ssh_private_key,
        ssh_public_key,
    )
