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
from logging import DEBUG, ERROR, INFO, WARN
from os.path import isfile
from pathlib import Path
from queue import Queue
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
from flwr.common.constant import Status, SubStatus
from flwr.common.logger import (
    log,
    mirror_output_to_queue,
    restore_output,
    start_log_uploader,
    stop_log_uploader,
)
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    run_from_proto,
    run_status_to_proto,
)
from flwr.common.typing import RunStatus
from flwr.proto.run_pb2 import UpdateRunStatusRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    PullServerAppInputsRequest,
    PullServerAppInputsResponse,
    PushServerAppOutputsRequest,
)
from flwr.server.driver.grpc_driver import GrpcDriver
from flwr.server.run_serverapp import run as run_


def flwr_serverapp() -> None:
    """Run process-isolated Flower ServerApp."""
    # Capture stdout/stderr
    log_queue: Queue[Optional[str]] = Queue()
    mirror_output_to_queue(log_queue)

    parser = argparse.ArgumentParser(
        description="Run a Flower ServerApp",
    )
    parser.add_argument(
        "--superlink",
        type=str,
        help="Address of SuperLink's DriverAPI",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="When set, this process will start a single ServerApp "
        "for a pending Run. If no pending run the process will exit. ",
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

    log(INFO, "Starting Flower ServerApp")
    certificates = _try_obtain_certificates(args)

    log(
        DEBUG,
        "Staring isolated `ServerApp` connected to SuperLink DriverAPI at %s",
        args.superlink,
    )
    run_serverapp(
        superlink=args.superlink,
        log_queue=log_queue,
        run_once=args.run_once,
        flwr_dir_=args.flwr_dir,
        certificates=certificates,
    )

    # Restore stdout/stderr
    restore_output()


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


def run_serverapp(  # pylint: disable=R0914, disable=W0212
    superlink: str,
    log_queue: Queue[Optional[str]],
    run_once: bool,
    flwr_dir_: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower ServerApp process."""
    driver = GrpcDriver(
        serverappio_service_address=superlink,
        root_certificates=certificates,
    )

    # Resolve directory where FABs are installed
    flwr_dir = get_flwr_dir(flwr_dir_)
    log_uploader = None

    while True:

        try:
            # Pull ServerAppInputs from LinkState
            req = PullServerAppInputsRequest()
            res: PullServerAppInputsResponse = driver._stub.PullServerAppInputs(req)
            if not res.HasField("run"):
                sleep(3)
                run_status = None
                continue

            context = context_from_proto(res.context)
            run = run_from_proto(res.run)
            fab = fab_from_proto(res.fab)

            driver.init_run(run.run_id)

            # Start log uploader for this run
            log_uploader = start_log_uploader(
                log_queue=log_queue,
                node_id=0,
                run_id=run.run_id,
                stub=driver._stub,
            )

            log(DEBUG, "ServerApp process starts FAB installation.")
            install_from_fab(fab.content, flwr_dir=flwr_dir, skip_prompt=True)

            fab_id, fab_version = get_fab_metadata(fab.content)

            app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str, flwr_dir))
            config = get_project_config(app_path)

            # Obtain server app reference and the run config
            server_app_attr = config["tool"]["flwr"]["app"]["components"]["serverapp"]
            server_app_run_config = get_fused_config_from_dir(
                Path(app_path), run.override_config
            )

            # Update run_config in context
            context.run_config = server_app_run_config

            log(
                DEBUG,
                "Flower will load ServerApp `%s` in %s",
                server_app_attr,
                app_path,
            )

            # Change status to Running
            run_status_proto = run_status_to_proto(RunStatus(Status.RUNNING, "", ""))
            driver._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

            # Load and run the ServerApp with the Driver
            updated_context = run_(
                driver=driver,
                server_app_dir=app_path,
                server_app_attr=server_app_attr,
                context=context,
            )

            # Send resulting context
            context_proto = context_to_proto(updated_context)
            out_req = PushServerAppOutputsRequest(
                run_id=run.run_id, context=context_proto
            )
            _ = driver._stub.PushServerAppOutputs(out_req)

            run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")

        except Exception as ex:  # pylint: disable=broad-exception-caught
            exc_entity = "ServerApp"
            log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
            run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))

        finally:
            if run_status:
                run_status_proto = run_status_to_proto(run_status)
                driver._stub.UpdateRunStatus(
                    UpdateRunStatusRequest(
                        run_id=run.run_id, run_status=run_status_proto
                    )
                )

            # Stop log uploader for this run
            if log_uploader:
                stop_log_uploader(log_queue, log_uploader)
                log_uploader = None

        # Stop the loop if `flwr-serverapp` is expected to process a single run
        if run_once:
            break
