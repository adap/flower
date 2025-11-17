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
    ExecPluginType,
    Status,
    SubStatus,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.heartbeat import HeartbeatSender, get_grpc_app_heartbeat_fn
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
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PushAppOutputsRequest,
)
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetFederationOptionsRequest,
    GetFederationOptionsResponse,
    UpdateRunStatusRequest,
)
from flwr.proto.simulationio_pb2_grpc import SimulationIoStub
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.simulation.run_simulation import _run_simulation
from flwr.simulation.simulationio_connection import SimulationIoConnection
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.superexec.plugin import SimulationExecPlugin
from flwr.supercore.superexec.run_superexec import run_with_deprecation_warning


def flwr_simulation() -> None:
    """Run process-isolated Flower Simulation."""
    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    args = _parse_args_run_flwr_simulation().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-simulation` does not support TLS yet.",
        )

    # Disallow long-running `flwr-simulation` processes
    if args.token is None:
        run_with_deprecation_warning(
            cmd="flwr-simulation",
            plugin_type=ExecPluginType.SIMULATION,
            plugin_class=SimulationExecPlugin,
            stub_class=SimulationIoStub,
            appio_api_address=args.simulationio_api_address,
            flwr_dir=args.flwr_dir,
            parent_pid=args.parent_pid,
            warn_run_once=args.run_once,
        )
        return

    log(INFO, "Starting Flower Simulation")
    log(
        DEBUG,
        "Starting isolated `Simulation` connected to SuperLink SimulationAppIo API "
        "at %s",
        args.simulationio_api_address,
    )
    run_simulation_process(
        simulationio_api_address=args.simulationio_api_address,
        log_queue=log_queue,
        token=args.token,
        flwr_dir_=args.flwr_dir,
        certificates=None,
        parent_pid=args.parent_pid,
    )

    # Restore stdout/stderr
    restore_output()


def run_simulation_process(  # pylint: disable=R0913, R0914, R0915, R0917, W0212
    simulationio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    flwr_dir_: str | None = None,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower Simulation process."""
    # Start monitoring the parent process if a PID is provided
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    conn = SimulationIoConnection(
        simulationio_service_address=simulationio_api_address,
        root_certificates=certificates,
    )

    # Resolve directory where FABs are installed
    flwr_dir = get_flwr_dir(flwr_dir_)
    log_uploader = None
    heartbeat_sender = None
    run_status = None

    try:
        # Pull SimulationInputs from LinkState
        req = PullAppInputsRequest(token=token)
        res: PullAppInputsResponse = conn._stub.PullAppInputs(req)
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
        federation_options = config_record_from_proto(fed_opt_res.federation_options)

        # Unflatten underlying dict
        fed_opt = unflatten_dict({**federation_options})

        # Extract configs values of interest
        num_supernodes = fed_opt.get("num-supernodes")
        if num_supernodes is None:
            raise ValueError("Federation options expects `num-supernodes` to be set.")
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

        # Set up heartbeat sender
        heartbeat_fn = get_grpc_app_heartbeat_fn(
            conn._stub,
            run.run_id,
            failure_message="Heartbeat failed unexpectedly. The SuperLink could "
            "not find the provided run ID, or the run status is invalid.",
        )
        heartbeat_sender = HeartbeatSender(heartbeat_fn)
        heartbeat_sender.start()

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
            server_app_context=context,
            is_app=True,
            exit_event=EventType.FLWR_SIMULATION_RUN_LEAVE,
        )

        # Send resulting context
        context_proto = context_to_proto(updated_context)
        out_req = PushAppOutputsRequest(
            token=token, run_id=run.run_id, context=context_proto
        )
        _ = conn._stub.PushAppOutputs(out_req)

        run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")

    except Exception as ex:  # pylint: disable=broad-exception-caught
        exc_entity = "Simulation"
        log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
        run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))

    finally:
        # Stop heartbeat sender
        if heartbeat_sender:
            heartbeat_sender.stop()

        # Stop log uploader for this run and upload final logs
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        # Update run status
        if run_status:
            run_status_proto = run_status_to_proto(run_status)
            conn._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

        # Clean up the Context if it exists
        try:
            del updated_context
        except NameError:
            pass


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
    add_args_flwr_app_common(parser=parser)
    return parser
