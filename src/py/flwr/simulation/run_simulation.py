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
"""Flower Simulation."""

import argparse
import threading

import grpc

from flwr.common import EventType, event
from flwr.server.app import _register_exit_handlers, _run_driver_api_grpc
from flwr.server.driver.driver import Driver
from flwr.server.run_serverapp import run
from flwr.server.superlink.fleet.vce.vce_api import run_vce
from flwr.server.superlink.state import StateFactory


def run_simulation() -> None:
    """."""
    args = _parse_args_run_simulation().parse_args()

    # Initialize StateFactory
    state_factory = StateFactory(":flwr-in-memory-state:")

    # Start Driver API
    driver_server: grpc.Server = _run_driver_api_grpc(
        address="0.0.0.0:9091",
        state_factory=state_factory,
        certificates=None,
    )

    # spawn thread w/ superlink+VCE
    superlink_th = threading.Thread(
        target=run_vce,
        args=(
            args.num_clients,
            {"num_cpus": 1, "num_gpus": 0.0},
            args.client_app,
            args.dir,
            state_factory,
        ),
        daemon=False,
    )

    event(EventType.RUN_SUPERLINK_ENTER)
    superlink_th.start()

    # Initialize Driver
    driver = Driver(
        driver_service_address="0.0.0.0:9091",
        root_certificates=None,
    )

    # Launch server app
    run(args.server_app, driver, args.dir)

    _register_exit_handlers(
        grpc_servers=[driver_server],
        bckg_threads=[superlink_th],
        event_type=EventType.RUN_SUPERLINK_LEAVE,
    )


def _parse_args_run_simulation() -> argparse.ArgumentParser:
    """Parse flower-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower Simulation",
    )
    parser.add_argument(
        "--client-app",
        help="For example: `client:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--num-clients",  # TODO: or supernodes ?
        type=int,
        help="Number of clients in your simulation.",
    )
    parser.add_argument(
        "--server-app",
        help="For example: `server:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server app without HTTPS. By default, the app runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load Flower "
        "app from there."
        " Default: current working directory.",
    )

    return parser
