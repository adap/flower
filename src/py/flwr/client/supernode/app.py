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
from logging import DEBUG, INFO, WARN
from pathlib import Path
from typing import Callable, Optional

from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.common import EventType, event
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log
from flwr.common.object_ref import load_app, validate

from ..app import _start_client_internal


def run_supernode() -> None:
    """Run Flower SuperNode."""
    log(INFO, "Starting Flower SuperNode")

    event(EventType.RUN_SUPERNODE_ENTER)

    _ = _parse_args_run_supernode().parse_args()

    log(
        DEBUG,
        "Flower SuperNode starting...",
    )

    # Graceful shutdown
    register_exit_handlers(
        event_type=EventType.RUN_SUPERNODE_LEAVE,
    )


def run_client_app() -> None:
    """Run Flower client app."""
    log(INFO, "Long-running Flower client starting")

    event(EventType.RUN_CLIENT_APP_ENTER)

    args = _parse_args_run_client_app().parse_args()

    root_certificates = _get_certificates(args)
    log(
        DEBUG,
        "Flower will load ClientApp `%s`",
        getattr(args, "client-app"),
    )
    load_fn = _get_load_client_app_fn(args)

    _start_client_internal(
        server_address=args.server,
        load_client_app_fn=load_fn,
        transport="rest" if args.rest else "grpc-rere",
        root_certificates=root_certificates,
        insecure=args.insecure,
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
    )
    register_exit_handlers(event_type=EventType.RUN_CLIENT_APP_LEAVE)


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
            args.server,
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
            args.server,
            cert_path,
        )
    return root_certificates


def _get_load_client_app_fn(
    args: argparse.Namespace,
) -> Callable[[], ClientApp]:
    """Get the load_client_app_fn function."""
    client_app_dir = args.dir
    if client_app_dir is not None:
        sys.path.insert(0, client_app_dir)

    app_ref: str = getattr(args, "client-app")
    valid, error_msg = validate(app_ref)
    if not valid and error_msg:
        raise LoadClientAppError(error_msg) from None

    def _load() -> ClientApp:
        client_app = load_app(app_ref, LoadClientAppError)

        if not isinstance(client_app, ClientApp):
            raise LoadClientAppError(
                f"Attribute {app_ref} is not of type {ClientApp}",
            ) from None

        return client_app

    return _load


def _parse_args_run_supernode() -> argparse.ArgumentParser:
    """Parse flower-supernode command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperNode",
    )

    parser.add_argument(
        "client-app",
        nargs="?",
        default="",
        help="For example: `client:app` or `project.package.module:wrapper.app`. "
        "This is optional and serves as the default ClientApp to be loaded when "
        "the ServerApp does not specify `fab_id` and `fab_version`. "
        "If not provided, defaults to an empty string.",
    )
    _parse_args_common(parser)
    parser.add_argument(
        "--flwr-dir",
        default=None,
        help="""The path containing installed Flower Apps.
    By default, this value isequal to:

        - `$FLWR_HOME/` if `$FLWR_HOME` is defined
        - `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined
        - `$HOME/.flwr/` in all other cases
    """,
    )

    return parser


def _parse_args_run_client_app() -> argparse.ArgumentParser:
    """Parse flower-client-app command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower client app",
    )

    parser.add_argument(
        "client-app",
        help="For example: `client:app` or `project.package.module:wrapper.app`",
    )
    _parse_args_common(parser=parser)

    return parser


def _parse_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the client without HTTPS. By default, the client runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--rest",
        action="store_true",
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
        default="0.0.0.0:9092",
        help="Server address",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="The maximum number of times the client will try to connect to the"
        "server before giving up in case of a connection error. By default,"
        "it is set to None, meaning there is no limit to the number of tries.",
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=None,
        help="The maximum duration before the client stops trying to"
        "connect to the server in case of connection error. By default, it"
        "is set to None, meaning there is no limit to the total time.",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load Flower "
        "app from there."
        " Default: current working directory.",
    )
