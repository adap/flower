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
"""Flower ServerApp process."""

import argparse
import sys
from logging import DEBUG, INFO, WARN
from os.path import isfile
from pathlib import Path
from typing import Optional

from flwr.common.logger import log
from flwr.server.driver.grpc_driver import GrpcDriver


def flwr_serverapp() -> None:
    """Run process-isolated Flower ServerApp."""
    log(INFO, "Starting Flower ServerApp")

    parser = argparse.ArgumentParser(
        description="Run a Flower ServerApp",
    )
    parser.add_argument(
        "--superlink",
        type=str,
        help="Address of SuperLink's DriverAPI",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=False,
        help="Id of the Run this process should start. If not supplied, this "
        "function will request a pending run to the LinkState.",
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
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server without HTTPS, regardless of whether certificate "
        "paths are provided. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--root-certificates",
        metavar="ROOT_CERT",
        type=str,
        help="Specifies the path to the PEM-encoded root certificate file for "
        "establishing secure HTTPS connections.",
    )
    args = parser.parse_args()

    certificates = _try_obtain_certificates(args)

    log(
        DEBUG,
        "Staring isolated `ServerApp` connected to SuperLink DriverAPI at %s "
        "for run-id %s",
        args.superlink,
        args.run_id,
    )
    run_serverapp(
        superlink=args.superlink,
        run_id=args.run_id,
        flwr_dir_=args.flwr_dir,
        certificates=certificates,
    )


def _try_obtain_certificates(
    args: argparse.Namespace,
) -> Optional[bytes]:

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
            "Option `--insecure` was set. Starting insecure HTTP channel to %s.",
            args.superlink,
        )
        root_certificates = None
    else:
        # Load the certificates if provided, or load the system certificates
        if not isfile(args.root_certificates):
            sys.exit("Path argument `--root-certificates` does not point to a file.")
        root_certificates = Path(args.root_certificates).read_bytes()
        log(
            DEBUG,
            "Starting secure HTTPS channel to %s "
            "with the following certificates: %s.",
            args.superlink,
            args.root_certificates,
        )
    return root_certificates


def run_serverapp(  # pylint: disable=R0914
    superlink: str,
    run_id: Optional[int] = None,
    flwr_dir_: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower ServerApp process."""
    _ = GrpcDriver(
        run_id=run_id if run_id else 0,
        driver_service_address=superlink,
        root_certificates=certificates,
    )

    log(INFO, "%s", flwr_dir_)

    # Then, GetServerInputs

    # Then, run ServerApp
