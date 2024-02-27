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
import asyncio
import threading
from typing import Optional

import grpc

from flwr.client import ClientApp
from flwr.common import EventType, event
from flwr.server.driver.driver import Driver
from flwr.server.run_serverapp import run
from flwr.server.server_app import ServerApp
from flwr.server.superlink.state import StateFactory


def run_simulation_from_cli() -> None:
    """."""
    args = _parse_args_run_simulation().parse_args()

    run_simulation(
        num_supernodes=args.num_supernodes,
        client_app_module_name=args.client_app,
        backend_name=args.backend,
        backend_config=args.backend_config,
        working_dir=args.dir,
        server_app_module_name=args.server_app,
    )


def run_simulation(
    num_supernodes: int,
    server_app: Optional[ServerApp] = None,
    client_app: Optional[ClientApp] = None,
    backend_name: str = "ray",
    backend_config: str = "{}",
    client_app_module_name: Optional[str] = None,
    server_app_module_name: Optional[str] = None,
    working_dir: str = "",
) -> None:
    """."""
    # TODO: below create circular imports
    from flwr.server.app import _register_exit_handlers, _run_driver_api_grpc
    from flwr.server.superlink.fleet.vce import start_vce

    # Initialize StateFactory
    state_factory = StateFactory(":flwr-in-memory-state:")

    # Start Driver API
    driver_address = "0.0.0.0:9098"
    driver_server: grpc.Server = _run_driver_api_grpc(
        address=driver_address,
        state_factory=state_factory,
        certificates=None,
    )

    # Superlink with Simulation Engine
    f_stop = asyncio.Event()
    superlink_th = threading.Thread(
        target=start_vce,
        kwargs={
            "num_supernodes": num_supernodes,
            "client_app_module_name": client_app_module_name,
            "client_app": client_app,
            "backend_name": backend_name,
            "backend_config_json_stream": backend_config,
            "working_dir": working_dir,
            "state_factory": state_factory,
            "f_stop": f_stop,
        },
        daemon=False,
    )

    event(EventType.RUN_SUPERLINK_ENTER)
    superlink_th.start()

    # Initialize Driver
    driver = Driver(
        driver_service_address=driver_address,
        root_certificates=None,
    )

    # Launch server app
    run(server_app_module_name, driver, working_dir, loaded_server_app=server_app)

    del driver

    # Trigger stop event
    f_stop.set()

    _register_exit_handlers(
        grpc_servers=[driver_server],
        bckg_threads=[superlink_th],
        event_type=EventType.RUN_SUPERLINK_LEAVE,
    )
    superlink_th.join()


def _parse_args_run_simulation() -> argparse.ArgumentParser:
    """Parse flower-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower Simulation",
    )
    parser.add_argument(
        "--client-app",
        required=True,
        help="For example: `client:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--server-app",
        required=True,
        help="For example: `server:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--num-supernodes",
        type=int,
        required=True,
        help="Number of simulated SuperNodes.",
    )
    parser.add_argument(
        "--backend",
        default="ray",
        type=str,
        help="Simulation backend that executes the ClientApp.",
    )
    parser.add_argument(
        "--backend-config",
        type=str,
        default='{"client_resources": {"num_cpus":2, "num_gpus":0.0}, "tensorflow": 0}',
        help='A JSON formatted stream, e.g \'{"<keyA>":<value>, "<keyB>":<value>}\' to '
        "configure a backend. Values supported in <value> are those included by "
        "`flwr.common.typing.ConfigsRecordValues`. ",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load"
        "ClientApp and ServerApp from there."
        " Default: current working directory.",
    )

    return parser
