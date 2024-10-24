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
from logging import DEBUG, INFO
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
    args = parser.parse_args()

    log(
        DEBUG,
        "Staring isolated `ServerApp` connected to SuperLink DriverAPI at %s "
        "for run-id %s",
        args.superlink,
        args.run_id,
    )
    run_serverapp(superlink=args.superlink, run_id=args.run_id)


def run_serverapp(  # pylint: disable=R0914
    superlink: str,
    run_id: Optional[int] = None,
) -> None:
    """Run Flower ServerApp process.

    Parameters
    ----------
    superlink : str
        Address of SuperLink
    run_id : Optional[int] (default: None)
        Unique identifier of a Run registered at the LinkState. If not supplied,
        this function will request a pending run to the LinkState.
    """
    _ = GrpcDriver(
        run_id=run_id if run_id else 0,
        driver_service_address=superlink,
        root_certificates=None,
    )

    # Then, GetServerInputs

    # Then, run ServerApp
