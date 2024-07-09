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
"""Run ServerApp."""


import argparse
import sys
from logging import DEBUG, INFO, WARN
from pathlib import Path
from typing import Dict, Optional

from flwr.common import Context, EventType, RecordSet, event
from flwr.common.config import (
    get_flwr_dir,
    get_fused_config,
    get_project_config,
    get_project_dir,
)
from flwr.common.logger import log, update_console_handler, warn_deprecated_feature
from flwr.common.object_ref import load_app
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
)

from .driver import Driver
from .driver.grpc_driver import GrpcDriver
from .server_app import LoadServerAppError, ServerApp

ADDRESS_DRIVER_API = "0.0.0.0:9091"


def run(
    driver: Driver,
    server_app_dir: str,
    server_config: Dict[str, str],
    server_app_attr: Optional[str] = None,
    loaded_server_app: Optional[ServerApp] = None,
) -> None:
    """Run ServerApp with a given Driver."""
    if not (server_app_attr is None) ^ (loaded_server_app is None):
        raise ValueError(
            "Either `server_app_attr` or `loaded_server_app` should be set "
            "but not both."
        )

    if server_app_dir is not None:
        sys.path.insert(0, str(Path(server_app_dir).absolute()))

    # Load ServerApp if needed
    def _load() -> ServerApp:
        if server_app_attr:
            server_app: ServerApp = load_app(
                server_app_attr, LoadServerAppError, server_app_dir
            )

            if not isinstance(server_app, ServerApp):
                raise LoadServerAppError(
                    f"Attribute {server_app_attr} is not of type {ServerApp}",
                ) from None

        if loaded_server_app:
            server_app = loaded_server_app
        return server_app

    server_app = _load()

    # Initialize Context
    context = Context(state=RecordSet(), run_config={})

    # Call ServerApp
    server_app(driver=driver, context=context)

    log(DEBUG, "ServerApp finished running.")


def run_server_app() -> None:  # pylint: disable=too-many-branches
    """Run Flower server app."""
    event(EventType.RUN_SERVER_APP_ENTER)

    args = _parse_args_run_server_app().parse_args()

    if args.server != ADDRESS_DRIVER_API:
        warn = "Passing flag --server is deprecated. Use --superlink instead."
        warn_deprecated_feature(warn)

        if args.superlink != ADDRESS_DRIVER_API:
            # if `--superlink` also passed, then
            # warn user that this argument overrides what was passed with `--server`
            log(
                WARN,
                "Both `--server` and `--superlink` were passed. "
                "`--server` will be ignored. Connecting to the Superlink Driver API "
                "at %s.",
                args.superlink,
            )
        else:
            args.superlink = args.server

    update_console_handler(
        level=DEBUG if args.verbose else INFO,
        timestamps=args.verbose,
        colored=True,
    )

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

    server_app_attr: Optional[str] = getattr(args, "server-app")
    if not (server_app_attr is None) ^ (args.run_id is None):
        raise sys.exit(
            "Please provide either a ServerApp reference or a Run ID, but not both. "
            "For more details, use: ``flower-server-app -h``"
        )

    # Initialize GrpcDriver
    if args.run_id is not None:
        # User provided `--run-id`, but not `server-app`
        driver = GrpcDriver(
            run_id=args.run_id,
            driver_service_address=args.superlink,
            root_certificates=root_certificates,
        )
    else:
        # User provided `server-app`, but not `--run-id`
        # Create run if run_id is not provided
        driver = GrpcDriver(
            run_id=0,  # Will be overwritten
            driver_service_address=args.superlink,
            root_certificates=root_certificates,
        )
        # Create run
        req = CreateRunRequest(fab_id=args.fab_id, fab_version=args.fab_version)
        res: CreateRunResponse = driver._stub.CreateRun(req)  # pylint: disable=W0212
        # Overwrite driver._run_id
        driver._run_id = res.run_id  # pylint: disable=W0212

    server_config = {}

    # Dynamically obtain ServerApp path based on run_id
    if args.run_id is not None:
        # User provided `--run-id`, but not `server-app`
        flwr_dir = get_flwr_dir(args.flwr_dir)
        run_ = driver.run
        server_app_dir = str(get_project_dir(run_.fab_id, run_.fab_version, flwr_dir))
        config = get_project_config(server_app_dir)
        server_app_attr = config["flower"]["components"]["serverapp"]
        server_config = get_fused_config(run_, flwr_dir)
    else:
        # User provided `server-app`, but not `--run-id`
        server_app_dir = str(Path(args.dir).absolute())

    log(DEBUG, "Flower will load ServerApp `%s` in %s", server_app_attr, server_app_dir)

    log(
        DEBUG,
        "root_certificates: `%s`",
        root_certificates,
    )

    # Run the ServerApp with the Driver
    run(
        driver=driver,
        server_app_dir=server_app_dir,
        server_config=server_config,
        server_app_attr=server_app_attr,
    )

    # Clean up
    driver.close()

    event(EventType.RUN_SERVER_APP_LEAVE)


def _parse_args_run_server_app() -> argparse.ArgumentParser:
    """Parse flower-server-app command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower server app",
    )

    parser.add_argument(
        "server-app",
        nargs="?",
        default=None,
        help="For example: `server:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server app without HTTPS. By default, the app runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set the logging to `DEBUG`.",
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
        default=ADDRESS_DRIVER_API,
        help="Server address",
    )
    parser.add_argument(
        "--superlink",
        default=ADDRESS_DRIVER_API,
        help="SuperLink Driver API (gRPC-rere) address (IPv4, IPv6, or a domain name)",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load Flower "
        "app from there."
        " Default: current working directory.",
    )
    parser.add_argument(
        "--fab-id",
        default=None,
        type=str,
        help="The identifier of the FAB used in the run.",
    )
    parser.add_argument(
        "--fab-version",
        default=None,
        type=str,
        help="The version of the FAB used in the run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        type=int,
        help="The identifier of the run.",
    )
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

    return parser
