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
"""Flower ServerApp process."""


import argparse
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from queue import Queue
from time import sleep
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common.args import add_args_flwr_app_common
from flwr.common.config import (
    get_flwr_dir,
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    Status,
    SubStatus,
)
from flwr.common.exit import ExitCode, flwr_exit
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
from flwr.common.telemetry import EventType, event
from flwr.common.typing import RunNotRunningException, RunStatus
from flwr.proto.run_pb2 import UpdateRunStatusRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    PullServerAppInputsRequest,
    PullServerAppInputsResponse,
    PushServerAppOutputsRequest,
)
from flwr.server.grid.grpc_grid import GrpcGrid
from flwr.server.run_serverapp import run as run_


def flwr_serverapp() -> None:
    """Run process-isolated Flower ServerApp."""
    # Capture stdout/stderr
    log_queue: Queue[Optional[str]] = Queue()
    mirror_output_to_queue(log_queue)

    args = _parse_args_run_flwr_serverapp().parse_args()

    log(INFO, "Start `flwr-serverapp` process")

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-serverapp` does not support TLS yet.",
        )

    log(
        DEBUG,
        "`flwr-serverapp` will attempt to connect to SuperLink's "
        "ServerAppIo API at %s",
        args.serverappio_api_address,
    )
    run_serverapp(
        serverappio_api_address=args.serverappio_api_address,
        log_queue=log_queue,
        run_once=args.run_once,
        flwr_dir=args.flwr_dir,
        certificates=None,
    )

    # Restore stdout/stderr
    restore_output()


def run_serverapp(  # pylint: disable=R0914, disable=W0212, disable=R0915
    serverappio_api_address: str,
    log_queue: Queue[Optional[str]],
    run_once: bool,
    flwr_dir: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower ServerApp process."""
    grid = GrpcGrid(
        serverappio_service_address=serverappio_api_address,
        root_certificates=certificates,
    )

    # Resolve directory where FABs are installed
    flwr_dir_ = get_flwr_dir(flwr_dir)
    log_uploader = None
    success = True
    hash_run_id = None
    run_status = None
    while True:

        try:
            # Pull ServerAppInputs from LinkState
            req = PullServerAppInputsRequest()
            log(DEBUG, "[flwr-serverapp] Pull ServerAppInputs")
            res: PullServerAppInputsResponse = grid._stub.PullServerAppInputs(req)
            if not res.HasField("run"):
                sleep(3)
                run_status = None
                continue

            context = context_from_proto(res.context)
            run = run_from_proto(res.run)
            fab = fab_from_proto(res.fab)

            hash_run_id = get_sha256_hash(run.run_id)

            grid.set_run(run.run_id)

            # Start log uploader for this run
            log_uploader = start_log_uploader(
                log_queue=log_queue,
                node_id=0,
                run_id=run.run_id,
                stub=grid._stub,
            )

            log(DEBUG, "[flwr-serverapp] Start FAB installation.")
            install_from_fab(fab.content, flwr_dir=flwr_dir_, skip_prompt=True)

            fab_id, fab_version = get_fab_metadata(fab.content)

            app_path = str(
                get_project_dir(fab_id, fab_version, fab.hash_str, flwr_dir_)
            )
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
                "[flwr-serverapp] Will load ServerApp `%s` in %s",
                server_app_attr,
                app_path,
            )

            # Change status to Running
            run_status_proto = run_status_to_proto(RunStatus(Status.RUNNING, "", ""))
            grid._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

            event(
                EventType.FLWR_SERVERAPP_RUN_ENTER,
                event_details={"run-id-hash": hash_run_id},
            )

            # Load and run the ServerApp with the Grid
            updated_context = run_(
                grid=grid,
                server_app_dir=app_path,
                server_app_attr=server_app_attr,
                context=context,
            )

            # Send resulting context
            context_proto = context_to_proto(updated_context)
            log(DEBUG, "[flwr-serverapp] Will push ServerAppOutputs")
            out_req = PushServerAppOutputsRequest(
                run_id=run.run_id, context=context_proto
            )
            _ = grid._stub.PushServerAppOutputs(out_req)

            run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
        except RunNotRunningException:
            log(INFO, "")
            log(INFO, "Run ID %s stopped.", run.run_id)
            log(INFO, "")
            run_status = None
            success = False

        except Exception as ex:  # pylint: disable=broad-exception-caught
            exc_entity = "ServerApp"
            log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
            run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))
            success = False

        finally:
            # Stop log uploader for this run and upload final logs
            if log_uploader:
                stop_log_uploader(log_queue, log_uploader)
                log_uploader = None

            # Update run status
            if run_status:
                run_status_proto = run_status_to_proto(run_status)
                grid._stub.UpdateRunStatus(
                    UpdateRunStatusRequest(
                        run_id=run.run_id, run_status=run_status_proto
                    )
                )
            event(
                EventType.FLWR_SERVERAPP_RUN_LEAVE,
                event_details={"run-id-hash": hash_run_id, "success": success},
            )

        # Stop the loop if `flwr-serverapp` is expected to process a single run
        if run_once:
            break


def _parse_args_run_flwr_serverapp() -> argparse.ArgumentParser:
    """Parse flwr-serverapp command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower ServerApp",
    )
    parser.add_argument(
        "--serverappio-api-address",
        default=SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's ServerAppIo API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="When set, this process will start a single ServerApp for a pending Run. "
        "If there is no pending Run, the process will exit.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
