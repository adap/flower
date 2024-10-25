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
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from time import sleep
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.config import (
    get_flwr_dir,
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    run_from_proto,
)
from flwr.proto.driver_pb2 import (
    PullServerAppInputsRequest,
    PullServerAppInputsResponse,
    PushServerAppOutputsRequest,
)
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
    args = parser.parse_args()

    log(
        DEBUG,
        "Staring isolated `ServerApp` connected to SuperLink DriverAPI at %s "
        "for run-id %s",
        args.superlink,
        args.run_id,
    )
    run_serverapp(superlink=args.superlink, run_id=args.run_id, flwr_dir=args.flwr_dir)


def run_serverapp(  # pylint: disable=R0914
    superlink: str,
    run_id: Optional[int] = None,
    flwr_dir: Optional[str] = None,
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
    driver = GrpcDriver(
        run_id=run_id if run_id else 0,
        driver_service_address=superlink,
        root_certificates=None,
    )

    # Resolve directory where FABs are installed
    flwr_dir = get_flwr_dir(flwr_dir)

    only_once = run_id is not None

    while True:

        try:
            # Pull ServerAppInputs from LinkState
            req = PullServerAppInputsRequest(run_id=run_id)
            res: PullServerAppInputsResponse = driver._stub.PullServerAppInputs(req)
            if not res.HasField("run"):
                sleep(3)
                continue

            context = context_from_proto(res.context)
            run = run_from_proto(res.run)
            fab = fab_from_proto(res.fab)

            log(DEBUG, "ServerApp process starts FAB installation.")
            install_from_fab(fab.content, flwr_dir=flwr_dir, skip_prompt=True)

            fab_id, fab_version = get_fab_metadata(fab.content)

            app_path = str(get_project_dir(fab_id, fab_version, fab.fab_hash, flwr_dir))
            config = get_project_config(app_path)

            # Obtain server app reference and the run config
            server_app_attr = config["tool"]["flwr"]["app"]["components"]["serverapp"]
            server_app_run_config = get_fused_config_from_dir(
                Path(app_path), driver.run.override_config
            )

            log(
                DEBUG,
                "Flower will load ServerApp `%s` in %s",
                server_app_attr,
                app_path,
            )

            # Load and run the ServerApp with the Driver
            updated_context = run(
                driver=driver,
                server_app_dir=app_path,
                server_app_run_config=server_app_run_config,
                server_app_attr=server_app_attr,
                context=context,
            )

            # Send resulting context
            context_proto = context_to_proto(updated_context)
            req = PushServerAppOutputsRequest(run_id=run_id, context=context_proto)
            res: PullServerAppInputsResponse = driver._stub.PushServerAppOutputs(req)

        except Exception as ex:  # pylint: disable=broad-exception-caught
            exc_entity = "ServerApp"
            log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)

        # Stop the loop if `flwr-serverapp` is expected to process a single run
        if only_once:
            break

        # Reset the run_id
        run_id = None
