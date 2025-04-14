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
"""Flower Simulation process."""


import argparse
from logging import DEBUG, ERROR, INFO
from queue import Queue
from time import sleep
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common import EventType, event
from flwr.common.args import add_args_flwr_app_common
from flwr.common.config import (
    get_flwr_dir,
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
    unflatten_dict,
)
from flwr.common.constant import (
    SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS,
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
    config_record_from_proto,
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    run_from_proto,
    run_status_to_proto,
)
from flwr.common.typing import RunStatus
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetFederationOptionsRequest,
    GetFederationOptionsResponse,
    UpdateRunStatusRequest,
)
from flwr.proto.simulationio_pb2 import (  # pylint: disable=E0611
    PullSimulationInputsRequest,
    PullSimulationInputsResponse,
    PushSimulationOutputsRequest,
)
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.simulation.run_simulation import _run_simulation
from flwr.simulation.simulationio_connection import SimulationIoConnection


def flwr_simulation() -> None:
    """Run process-isolated Flower Simulation."""
    # Capture stdout/stderr
    log_queue: Queue[Optional[str]] = Queue()
    mirror_output_to_queue(log_queue)

    args = _parse_args_run_flwr_simulation().parse_args()

    log(INFO, "Starting Flower Simulation")

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-simulation` does not support TLS yet. ",
        )

    log(
        DEBUG,
        "Starting isolated `Simulation` connected to SuperLink SimulationAppIo API "
        "at %s",
        args.simulationio_api_address,
    )
    run_simulation_process(
        simulationio_api_address=args.simulationio_api_address,
        log_queue=log_queue,
        run_once=args.run_once,
        flwr_dir_=args.flwr_dir,
        certificates=None,
    )

    # Restore stdout/stderr
    restore_output()


def run_simulation_process(  # pylint: disable=R0914, disable=W0212, disable=R0915
    simulationio_api_address: str,
    log_queue: Queue[Optional[str]],
    run_once: bool,
    flwr_dir_: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower Simulation process."""
    conn = SimulationIoConnection(
        simulationio_service_address=simulationio_api_address,
        root_certificates=certificates,
    )

    # Resolve directory where FABs are installed
    flwr_dir = get_flwr_dir(flwr_dir_)
    log_uploader = None

    while True:

        try:
            # Pull SimulationInputs from LinkState
            req = PullSimulationInputsRequest()
            res: PullSimulationInputsResponse = conn._stub.PullSimulationInputs(req)
            if not res.HasField("run"):
                sleep(3)
                run_status = None
                continue

            context = context_from_proto(res.context)
            run = run_from_proto(res.run)
            fab = fab_from_proto(res.fab)

            # Start log uploader for this run
            log_uploader = start_log_uploader(
                log_queue=log_queue,
                node_id=context.node_id,
                run_id=run.run_id,
                stub=conn._stub,
            )

            log(DEBUG, "Simulation process starts FAB installation.")
            install_from_fab(fab.content, flwr_dir=flwr_dir, skip_prompt=True)

            fab_id, fab_version = get_fab_metadata(fab.content)

            app_path = get_project_dir(fab_id, fab_version, fab.hash_str, flwr_dir)
            config = get_project_config(app_path)

            # Get ClientApp and SeverApp components
            app_components = config["tool"]["flwr"]["app"]["components"]
            client_app_attr = app_components["clientapp"]
            server_app_attr = app_components["serverapp"]
            fused_config = get_fused_config_from_dir(app_path, run.override_config)

            # Update run_config in context
            context.run_config = fused_config

            log(
                DEBUG,
                "Flower will load ServerApp `%s` in %s",
                server_app_attr,
                app_path,
            )
            log(
                DEBUG,
                "Flower will load ClientApp `%s` in %s",
                client_app_attr,
                app_path,
            )

            # Change status to Running
            run_status_proto = run_status_to_proto(RunStatus(Status.RUNNING, "", ""))
            conn._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

            # Pull Federation Options
            fed_opt_res: GetFederationOptionsResponse = conn._stub.GetFederationOptions(
                GetFederationOptionsRequest(run_id=run.run_id)
            )
            federation_options = config_record_from_proto(
                fed_opt_res.federation_options
            )

            # Unflatten underlying dict
            fed_opt = unflatten_dict({**federation_options})

            # Extract configs values of interest
            num_supernodes = fed_opt.get("num-supernodes")
            if num_supernodes is None:
                raise ValueError(
                    "Federation options expects `num-supernodes` to be set."
                )
            backend_config: BackendConfig = fed_opt.get("backend", {})
            verbose: bool = fed_opt.get("verbose", False)
            enable_tf_gpu_growth: bool = fed_opt.get("enable_tf_gpu_growth", False)

            event(
                EventType.FLWR_SIMULATION_RUN_ENTER,
                event_details={
                    "backend": "ray",
                    "num-supernodes": num_supernodes,
                    "run-id-hash": get_sha256_hash(run.run_id),
                },
            )

            # Launch the simulation
            updated_context = _run_simulation(
                server_app_attr=server_app_attr,
                client_app_attr=client_app_attr,
                num_supernodes=num_supernodes,
                backend_config=backend_config,
                app_dir=str(app_path),
                run=run,
                enable_tf_gpu_growth=enable_tf_gpu_growth,
                verbose_logging=verbose,
                server_app_run_config=fused_config,
                is_app=True,
                exit_event=EventType.FLWR_SIMULATION_RUN_LEAVE,
            )

            # Send resulting context
            context_proto = context_to_proto(updated_context)
            out_req = PushSimulationOutputsRequest(
                run_id=run.run_id, context=context_proto
            )
            _ = conn._stub.PushSimulationOutputs(out_req)

            run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")

        except Exception as ex:  # pylint: disable=broad-exception-caught
            exc_entity = "Simulation"
            log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
            run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))

        finally:
            # Stop log uploader for this run and upload final logs
            if log_uploader:
                stop_log_uploader(log_queue, log_uploader)
                log_uploader = None

            # Update run status
            if run_status:
                run_status_proto = run_status_to_proto(run_status)
                conn._stub.UpdateRunStatus(
                    UpdateRunStatusRequest(
                        run_id=run.run_id, run_status=run_status_proto
                    )
                )

        # Stop the loop if `flwr-simulation` is expected to process a single run
        if run_once:
            break


def _parse_args_run_flwr_simulation() -> argparse.ArgumentParser:
    """Parse flwr-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower Simulation",
    )
    parser.add_argument(
        "--simulationio-api-address",
        default=SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's SimulationIO API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="When set, this process will start a single simulation "
        "for a pending Run. If no pending run the process will exit. ",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
