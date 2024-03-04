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
from time import sleep
from typing import Any, Callable

import grpc

from flwr.common import EventType, event, log
from flwr.server.driver.driver import Driver
from flwr.server.run_serverapp import run
from flwr.server.superlink.driver.driver_grpc import run_driver_api_grpc
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.state import StateFactory
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth


def run_serverapp_th(
    server_app_attr: str,
    driver: Driver,
    server_app_dir: str,
    f_stop: asyncio.Event,
    delay_launch: int = 3,
) -> threading.Thread:
    """Run SeverApp in a thread."""
    serverapp_th = threading.Thread(
        target=run,
        kwargs={
            "server_app_attr": server_app_attr,
            "driver": driver,
            "server_app_dir": server_app_dir,
            "stop_event": f_stop,  # will be set when `run()` finishes
            # will trigger the shutdown of the Simulation Engine
        },
    )
    sleep(delay_launch)
    serverapp_th.start()
    return serverapp_th


def get_thread_exception_hook(stop_event: asyncio.Event) -> Callable[[Any], None]:
    """Return a callback for when the ServerApp thread raises an exception."""

    def execepthook(args: Any) -> None:
        """Upon exception raised, log exception and trigger stop event."""
        # log
        log(
            ERROR,
            "The ServerApp thread triggered exception (%s): %s",
            args.exc_type,
            args.exc_value,
        )
        log(ERROR, traceback.format_exc())
        # Set stop event
        stop_event.set()
        log(WARNING, "Triggered stop event for Simulation Engine.")

    return execepthook


def run_simulation() -> None:
    """Run Simulation Engine."""
    args = _parse_args_run_simulation().parse_args()

    # Load JSON config
    backend_config_dict = json.loads(args.backend_config)

    # Enable GPU memory growth (relevant only for TF)
    if args.enable_tf_gpu_growth:
        log(INFO, "Enabling GPU growth for Tensorflow on the main thread.")
        enable_tf_gpu_growth()
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config_dict.get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config_dict["tensorflow"] = True

    # Convert back to JSON stream
    backend_config = json.dumps(backend_config_dict)

    # Initialize StateFactory
    state_factory = StateFactory(":flwr-in-memory-state:")

    # Start Driver API
    driver_server: grpc.Server = run_driver_api_grpc(
        address=args.driver_api_address,
        state_factory=state_factory,
        certificates=None,
    )

    f_stop = asyncio.Event()
    serverapp_th = None
    try:
        # Initialize Driver
        driver = Driver(
            driver_service_address=args.driver_api_address,
            root_certificates=None,
        )

        # Get and run ServerApp thread
        serverapp_th = run_serverapp_th(args.server_app, driver, args.dir, f_stop)
        # Setup an exception hook
        threading.excepthook = get_thread_exception_hook(f_stop)

        # SuperLink with Simulation Engine
        event(EventType.RUN_SUPERLINK_ENTER)
        vce.start_vce(
            num_supernodes=args.num_supernodes,
            client_app_module_name=args.client_app,
            backend_name=args.backend,
            backend_config_json_stream=backend_config,
            working_dir=args.dir,
            state_factory=state_factory,
            f_stop=f_stop,
        )

    except Exception as ex:

        log(ERROR, "An exception occurred: %s", ex)
        log(ERROR, traceback.format_exc())
        raise RuntimeError("An error was encountered. Ending simulation.") from ex

    finally:

        # Stop Driver
        driver_server.stop(grace=0)
        del driver
        # Trigger stop event
        f_stop.set()

        event(EventType.RUN_SUPERLINK_LEAVE)
        if serverapp_th:
            serverapp_th.join()

    log(INFO, "Stopping Simulation Engine now.")


def _parse_args_run_simulation() -> argparse.ArgumentParser:
    """Parse flower-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower simulation",
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
        "out-of-memory error because TensorFlow by default allocates all GPU memory."
        "Read more about how `tf.config.experimental.set_memory_growth()` works in "
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
