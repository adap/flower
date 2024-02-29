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
import json
import threading
import traceback
from logging import ERROR, INFO, WARNING
from typing import Dict, Optional

import grpc

from flwr.client import ClientApp
from flwr.common import EventType, event, log
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.typing import ConfigsRecordValues
from flwr.server.driver.driver import Driver
from flwr.server.run_serverapp import run
from flwr.server.server_app import ServerApp
from flwr.server.superlink.driver.driver_grpc import run_driver_api_grpc
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.state import StateFactory
from flwr.simulation.ray_transport.utils import (
    enable_tf_gpu_growth as enable_gpu_growth,
)


def run_simulation_from_cli() -> None:
    """Run Simulation Engine from the CLI."""
    args = _parse_args_run_simulation().parse_args()

    # Load JSON config
    backend_config_dict = json.loads(args.backend_config)

    run_simulation(
        num_supernodes=args.num_supernodes,
        client_app_module_name=args.client_app,
        server_app_module_name=args.server_app,
        backend_name=args.backend,
        backend_config=backend_config_dict,
        working_dir=args.dir,
        driver_api_address=args.driver_api_address,
        enable_tf_gpu_growth=args.enable_tf_gpu_growth,
    )


# pylint: disable=too-many-arguments,too-many-locals
def run_simulation(
    num_supernodes: int,
    client_app: Optional[ClientApp] = None,
    server_app: Optional[ServerApp] = None,
    backend_name: str = "ray",
    backend_config: Optional[Dict[str, ConfigsRecordValues]] = None,
    client_app_module_name: Optional[str] = None,
    server_app_module_name: Optional[str] = None,
    working_dir: str = "",
    driver_api_address: str = "0.0.0.0:9091",
    enable_tf_gpu_growth: bool = False,
) -> None:
    r"""Launch the Simulation Engine.

    Parameters
    ----------
    num_supernodes : int
        Number of nodes that run a ClientApp. They can be sampled by a
        Driver in the ServerApp and receive a Message describing what the ClientApp
        should perform.

    client_app : Optional[ClientApp]
        The `ClientApp` to be executed by each of the `SuperNodes`. It will receive
        messages sent by the `ServerApp`.

    server_app : Optional[ServerApp]
        The `ServerApp` to be executed.

    backend_name : str (default: ray)
        A simulation backend that runs `ClientApp`s.

    backend_config : Optional[Dict[str, ConfigsRecordValues]]
        'A dictionary, e.g {"<keyA>":<value>, "<keyB>":<value>} to configure a
        backend. Values supported in <value> are those included by
        `flwr.common.typing.ConfigsRecordValues`.

    client_app_module_name : str
        A path to a `ClientApp` module to be loaded: For example: `client:app` or
        `project.package.module:wrapper.app`."

    server_app_module_name : str
        A path to a `ServerApp` module to be loaded: For example: `server:app` or
        `project.package.module:wrapper.app`."

    working_dir : str
        Add specified directory to the PYTHONPATH and load `ClientApp` from there.
        (Default: current working directory.)

    driver_api_address : str (default: "0.0.0.0:9091")
        Driver API (gRPC) server address (IPv4, IPv6, or a domain name)

    enable_tf_gpu_growth : bool (default: False)
        A boolean to indicate whether to enable GPU growth on the main thread. This is
        desirable if you make use of a TensorFlow model on your `ServerApp` while
        having your `ClientApp` running on the same GPU. Without enabling this, you
        might encounter an out-of-memory error becasue TensorFlow by default allocates
        all GPU memory. Read mor about how `tf.config.experimental.set_memory_growth()`
        works in the TensorFlow documentation: https://www.tensorflow.org/api/stable.
    """
    if backend_config is None:
        backend_config = {}

    # Enable GPU memory growth (relevant only for TF)
    if enable_tf_gpu_growth:
        log(INFO, "Enabling GPU growth for Tensorflow on the main thread.")
        enable_gpu_growth()
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config.get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config["tensorflow"] = True

    # Convert config to original JSON-stream format
    backend_config_stream = json.dumps(backend_config)

    # Initialize StateFactory
    state_factory = StateFactory(":flwr-in-memory-state-with-mapping:")

    # Start Driver API
    driver_server: grpc.Server = run_driver_api_grpc(
        address=driver_api_address,
        state_factory=state_factory,
        certificates=None,
    )

    # Superlink with Simulation Engine
    f_stop = asyncio.Event()
    superlink_th = threading.Thread(
        target=vce.start_vce,
        kwargs={
            "num_supernodes": num_supernodes,
            "client_app_module_name": client_app_module_name,
            "client_app": client_app,
            "backend_name": backend_name,
            "backend_config_json_stream": backend_config_stream,
            "working_dir": working_dir,
            "state_factory": state_factory,
            "f_stop": f_stop,
        },
        daemon=False,
    )

    superlink_th.start()
    event(EventType.RUN_SUPERLINK_ENTER)

    try:
        # Initialize Driver
        driver = Driver(
            driver_service_address=driver_api_address,
            root_certificates=None,
        )

        # Launch server app
        run(
            driver=driver,
            server_app_dir=working_dir,
            server_app_attr=server_app_module_name,
            loaded_server_app=server_app,
        )

    except Exception as ex:

        log(ERROR, "An exception occured !! %s", ex)
        log(ERROR, traceback.format_exc())
        raise RuntimeError(
            "An error was encountered by the Simulation Engine. Ending Simulation."
        ) from ex

    finally:

        del driver

        # Trigger stop event
        f_stop.set()

        register_exit_handlers(
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
        "--driver-api-address",
        default="0.0.0.0:9091",
        type=str,
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
        "--enable-tf-gpu-growth",
        action="store_true",
        help="Enables GPU growth on the main thread. This is desirable if you make "
        "use of a TensorFlow model on your `ServerApp` while having your `ClientApp` "
        "running on the same GPU. Without enabling this, you might encounter an "
        "out-of-memory error becasue TensorFlow by default allocates all GPU memory."
        "Read mor about how `tf.config.experimental.set_memory_growth()` works in "
        "the TensorFlow documentation: https://www.tensorflow.org/api/stable.",
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
