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
"""Flower SuperExec."""


import time
from typing import Optional

from flwr.common.config import get_flwr_dir
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import run_from_proto
from flwr.common.telemetry import EventType
from flwr.common.typing import Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    RequestTokenRequest,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.supercore.app_utils import start_parent_process_monitor

from .plugin import ExecPlugin


def run_superexec(
    plugin_class: type[ExecPlugin],
    stub_class: type[ClientAppIoStub],
    appio_api_address: str,
    flwr_dir: Optional[str] = None,
    parent_pid: Optional[int] = None,
) -> None:
    """Run Flower SuperExec.

    Parameters
    ----------
    plugin_class : type[ExecPlugin]
        The class of the SuperExec plugin to use.
    stub_class : type[ClientAppIoStub]
        The gRPC stub class for the AppIO API.
    appio_api_address : str
        The address of the AppIO API.
    flwr_dir : Optional[str] (default: None)
        The Flower directory.
    parent_pid : Optional[int] (default: None)
        The PID of the parent process. If provided, the SuperExec will terminate
        when the parent process exits.
    """
    # Start monitoring the parent process if a PID is provided
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Create the channel to the AppIO API
    # No TLS support for now, so insecure connection
    channel = create_channel(
        server_address=appio_api_address,
        insecure=True,
        root_certificates=None,
    )
    channel.subscribe(on_channel_state_change)

    # Register exit handlers to close the channel on exit
    register_exit_handlers(
        event_type=EventType.RUN_SUPEREXEC_LEAVE,
        exit_message="SuperExec terminated gracefully.",
        exit_handlers=[lambda: channel.close()],  # pylint: disable=W0108
    )

    # Create the gRPC stub for the AppIO API
    stub = stub_class(channel)
    _wrap_stub(stub, _make_simple_grpc_retry_invoker())

    def get_run(run_id: int) -> Run:
        _req = GetRunRequest(run_id=run_id)
        _res = stub.GetRun(_req)
        return run_from_proto(_res.run)

    # Create the SuperExec plugin instance
    plugin = plugin_class(
        appio_api_address=appio_api_address,
        flwr_dir=str(get_flwr_dir(flwr_dir)),
        get_run=get_run,
    )

    # Start the main loop
    try:
        while True:
            # Fetch suitable run IDs
            ls_req = ListAppsToLaunchRequest()
            ls_res = stub.ListAppsToLaunch(ls_req)

            # Allow the plugin to select a run ID
            run_id = None
            if ls_res.run_ids:
                run_id = plugin.select_run_id(candidate_run_ids=ls_res.run_ids)

            # Apply for a token if a run ID was selected
            if run_id is not None:
                tk_req = RequestTokenRequest(run_id=run_id)
                tk_res = stub.RequestToken(tk_req)

                # Launch the app if a token was granted; do nothing if not
                if tk_res.token:
                    plugin.launch_app(token=tk_res.token, run_id=run_id)

            # Sleep for a while before checking again
            time.sleep(1)
    finally:
        channel.close()
