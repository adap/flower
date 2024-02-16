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
from logging import DEBUG, WARN
from pathlib import Path

from flwr.common import EventType, event
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.recordset import RecordSet

from .driver.driver import Driver
from .serverapp import ServerApp, load_server_app


def run_server_app() -> None:
    """Run Flower server app."""
    event(EventType.RUN_SERVER_APP_ENTER)

    args = _parse_args_run_server_app().parse_args()

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

    log(
        DEBUG,
        "Flower will load ServerApp `%s`",
        getattr(args, "server-app"),
    )

    log(
        DEBUG,
        "root_certificates: `%s`",
        root_certificates,
    )

    log(WARN, "Not implemented: run_server_app")

    server_app_dir = args.dir
    if server_app_dir is not None:
        sys.path.insert(0, server_app_dir)

    def _load() -> ServerApp:
        server_app: ServerApp = load_server_app(getattr(args, "server-app"))
        return server_app

    server_app = _load()

    log(DEBUG, "server_app: `%s`", server_app)

    # Initialize Context
    context = Context(state=RecordSet())

    # Initialize Driver
    driver = Driver(
        driver_service_address=args.server,
        root_certificates=root_certificates,
    )

    # Call ServerApp
    server_app(driver=driver, context=context)

    # Clean up
    del driver

    event(EventType.RUN_SERVER_APP_LEAVE)


def _parse_args_run_server_app() -> argparse.ArgumentParser:
    """Parse flower-server-app command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower server app",
    )

    parser.add_argument(
        "server-app",
        help="For example: `server:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server app without HTTPS. By default, the app runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
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
        default="0.0.0.0:9091",
        help="Server address",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load Flower "
        "app from there."
        " Default: current working directory.",
    )

    return parser
