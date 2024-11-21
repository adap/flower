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
import logging
import sys
import threading
import traceback
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from time import sleep
from typing import Any, Optional

from flwr.cli.config_utils import load_and_validate
from flwr.client import ClientApp
from flwr.common import Context, EventType, RecordSet, event, log, now
from flwr.common.config import get_fused_config_from_dir, parse_config_args
from flwr.common.constant import RUN_ID_NUM_BYTES, Status
from flwr.common.logger import (
    set_logger_propagation,
    update_console_handler,
    warn_deprecated_feature_with_example,
)
from flwr.common.typing import Run, RunStatus, UserConfig
from flwr.server.driver import Driver, InMemoryDriver
from flwr.server.run_serverapp import run as _run
from flwr.server.server_app import ServerApp
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.linkstate.in_memory_linkstate import RunRecord
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.ray_transport.utils import (
    enable_tf_gpu_growth as enable_gpu_growth,
)


def _replace_keys(d: Any, match: str, target: str) -> Any:
    if isinstance(d, dict):
        return {
            k.replace(match, target): _replace_keys(v, match, target)
            for k, v in d.items()
        }
    if isinstance(d, list):
        return [_replace_keys(i, match, target) for i in d]
    return d


# Entry point from CLI
# pylint: disable=too-many-locals
def run_simulation_from_cli() -> None:
    """Run Simulation Engine from the CLI."""
    args = _parse_args_run_simulation().parse_args()

    event(
        EventType.CLI_FLOWER_SIMULATION_ENTER,
        event_details={"backend": args.backend, "num-supernodes": args.num_supernodes},
    )

    if args.enable_tf_gpu_growth:
        warn_deprecated_feature_with_example(
            "Passing `--enable-tf-gpu-growth` is deprecated.",
            example_message="Instead, set the `TF_FORCE_GPU_ALLOW_GROWTH` environmnet "
            "variable to true.",
            code_example='TF_FORCE_GPU_ALLOW_GROWTH="true" flower-simulation <...>',
        )

    # Load JSON config
    backend_config_dict = json.loads(args.backend_config)

    if backend_config_dict:
        # Backend config internally operates with `_` not with `-`
        backend_config_dict = _replace_keys(backend_config_dict, match="-", target="_")
        log(DEBUG, "backend_config_dict: %s", backend_config_dict)

    run_id = (
        generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
        if args.run_id is None
        else args.run_id
    )

    app_path = Path(args.app)
    if not app_path.is_dir():
        log(ERROR, "--app is not a directory")
        sys.exit("Simulation Engine cannot start.")

    # Load pyproject.toml
    config, errors, warnings = load_and_validate(
        app_path / "pyproject.toml", check_module=False
    )
    if errors:
        raise ValueError(errors)

    if warnings:
        log(WARNING, warnings)

    if config is None:
        raise ValueError("Config extracted from FAB's pyproject.toml is not valid")

    # Get ClientApp and SeverApp components
    app_components = config["tool"]["flwr"]["app"]["components"]
    client_app_attr = app_components["clientapp"]
    server_app_attr = app_components["serverapp"]

    override_config = parse_config_args(
        [args.run_config] if args.run_config else args.run_config
    )
    fused_config = get_fused_config_from_dir(app_path, override_config)

    # Create run
    run = Run.create_empty(run_id)
    run.override_config = override_config

    _run_simulation(
        server_app_attr=server_app_attr,
        client_app_attr=client_app_attr,
        num_supernodes=args.num_supernodes,
        backend_name=args.backend,
        backend_config=backend_config_dict,
        app_dir=args.app,
        run=run,
        enable_tf_gpu_growth=args.enable_tf_gpu_growth,
        delay_start=args.delay_start,
        verbose_logging=args.verbose,
        server_app_run_config=fused_config,
        is_app=True,
        exit_event=EventType.CLI_FLOWER_SIMULATION_LEAVE,
    )


# Entry point from Python session (script or notebook)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_simulation(
    server_app: ServerApp,
    client_app: ClientApp,
    num_supernodes: int,
    backend_name: str = "ray",
    backend_config: Optional[BackendConfig] = None,
    enable_tf_gpu_growth: bool = False,
    verbose_logging: bool = False,
) -> None:
    r"""Run a Flower App using the Simulation Engine.

    Parameters
    ----------
    server_app : ServerApp
        The `ServerApp` to be executed. It will send messages to different `ClientApp`
        instances running on different (virtual) SuperNodes.

    client_app : ClientApp
        The `ClientApp` to be executed by each of the SuperNodes. It will receive
        messages sent by the `ServerApp`.

    num_supernodes : int
        Number of nodes that run a ClientApp. They can be sampled by a Driver in the
        ServerApp and receive a Message describing what the ClientApp should perform.

    backend_name : str (default: ray)
        A simulation backend that runs `ClientApp`s.

    backend_config : Optional[BackendConfig]
        'A dictionary to configure a backend. Separate dictionaries to configure
        different elements of backend. Supported top-level keys are `init_args`
        for values parsed to initialisation of backend, `client_resources`
        to define the resources for clients, and `actor` to define the actor
        parameters. Values supported in <value> are those included by
        `flwr.common.typing.ConfigsRecordValues`.

    enable_tf_gpu_growth : bool (default: False)
        A boolean to indicate whether to enable GPU growth on the main thread. This is
        desirable if you make use of a TensorFlow model on your `ServerApp` while
        having your `ClientApp` running on the same GPU. Without enabling this, you
        might encounter an out-of-memory error because TensorFlow, by default, allocates
        all GPU memory. Read more about how `tf.config.experimental.set_memory_growth()`
        works in the TensorFlow documentation: https://www.tensorflow.org/api/stable.

    verbose_logging : bool (default: False)
        When disabled, only INFO, WARNING and ERROR log messages will be shown. If
        enabled, DEBUG-level logs will be displayed.
    """
    event(
        EventType.PYTHON_API_RUN_SIMULATION_ENTER,
        event_details={"backend": backend_name, "num-supernodes": num_supernodes},
    )

    if enable_tf_gpu_growth:
        warn_deprecated_feature_with_example(
            "Passing `enable_tf_gpu_growth=True` is deprecated.",
            example_message="Instead, set the `TF_FORCE_GPU_ALLOW_GROWTH` environment "
            "variable to true.",
            code_example='import os;os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"'
            "\n\tflwr.simulation.run_simulationt(...)",
        )

    _run_simulation(
        num_supernodes=num_supernodes,
        client_app=client_app,
        server_app=server_app,
        backend_name=backend_name,
        backend_config=backend_config,
        enable_tf_gpu_growth=enable_tf_gpu_growth,
        verbose_logging=verbose_logging,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_serverapp_th(
    server_app_attr: Optional[str],
    server_app: Optional[ServerApp],
    server_app_run_config: UserConfig,
    driver: Driver,
    app_dir: str,
    f_stop: threading.Event,
    has_exception: threading.Event,
    enable_tf_gpu_growth: bool,
    run_id: int,
) -> threading.Thread:
    """Run SeverApp in a thread."""

    def server_th_with_start_checks(
        tf_gpu_growth: bool,
        stop_event: threading.Event,
        exception_event: threading.Event,
        _driver: Driver,
        _server_app_dir: str,
        _server_app_run_config: UserConfig,
        _server_app_attr: Optional[str],
        _server_app: Optional[ServerApp],
    ) -> None:
        """Run SeverApp, after check if GPU memory growth has to be set.

        Upon exception, trigger stop event for Simulation Engine.
        """
        try:
            if tf_gpu_growth:
                log(INFO, "Enabling GPU growth for Tensorflow on the server thread.")
                enable_gpu_growth()

            # Initialize Context
            context = Context(
                run_id=run_id,
                node_id=0,
                node_config={},
                state=RecordSet(),
                run_config=_server_app_run_config,
            )

            # Run ServerApp
            _run(
                driver=_driver,
                context=context,
                server_app_dir=_server_app_dir,
                server_app_attr=_server_app_attr,
                loaded_server_app=_server_app,
            )
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "ServerApp thread raised an exception: %s", ex)
            log(ERROR, traceback.format_exc())
            exception_event.set()
            raise
        finally:
            log(DEBUG, "ServerApp finished running.")
            # Upon completion, trigger stop event if one was passed
            if stop_event is not None:
                stop_event.set()
                log(DEBUG, "Triggered stop event for Simulation Engine.")

    serverapp_th = threading.Thread(
        target=server_th_with_start_checks,
        args=(
            enable_tf_gpu_growth,
            f_stop,
            has_exception,
            driver,
            app_dir,
            server_app_run_config,
            server_app_attr,
            server_app,
        ),
    )
    serverapp_th.start()
    return serverapp_th


# pylint: disable=too-many-locals,too-many-positional-arguments
def _main_loop(
    num_supernodes: int,
    backend_name: str,
    backend_config_stream: str,
    app_dir: str,
    is_app: bool,
    enable_tf_gpu_growth: bool,
    run: Run,
    exit_event: EventType,
    delay_start: int,
    flwr_dir: Optional[str] = None,
    client_app: Optional[ClientApp] = None,
    client_app_attr: Optional[str] = None,
    server_app: Optional[ServerApp] = None,
    server_app_attr: Optional[str] = None,
    server_app_run_config: Optional[UserConfig] = None,
) -> None:
    """Start ServerApp on a separate thread, then launch Simulation Engine."""
    # Initialize StateFactory
    state_factory = LinkStateFactory(":flwr-in-memory-state:")

    f_stop = threading.Event()
    # A Threading event to indicate if an exception was raised in the ServerApp thread
    server_app_thread_has_exception = threading.Event()
    serverapp_th = None
    success = True
    try:
        # Register run
        log(DEBUG, "Pre-registering run with id %s", run.run_id)
        run.status = RunStatus(Status.RUNNING, "", "")
        run.starting_at = now().isoformat()
        run.running_at = run.starting_at
        state_factory.state().run_ids[run.run_id] = RunRecord(run=run)  # type: ignore

        if server_app_run_config is None:
            server_app_run_config = {}

        # Initialize Driver
        driver = InMemoryDriver(state_factory=state_factory)
        driver.set_run(run_id=run.run_id)

        # Get and run ServerApp thread
        serverapp_th = run_serverapp_th(
            server_app_attr=server_app_attr,
            server_app=server_app,
            server_app_run_config=server_app_run_config,
            driver=driver,
            app_dir=app_dir,
            f_stop=f_stop,
            has_exception=server_app_thread_has_exception,
            enable_tf_gpu_growth=enable_tf_gpu_growth,
            run_id=run.run_id,
        )

        # Buffer time so the `ServerApp` in separate thread is ready
        log(DEBUG, "Buffer time delay: %ds", delay_start)
        sleep(delay_start)
        # Start Simulation Engine
        vce.start_vce(
            num_supernodes=num_supernodes,
            client_app_attr=client_app_attr,
            client_app=client_app,
            backend_name=backend_name,
            backend_config_json_stream=backend_config_stream,
            app_dir=app_dir,
            is_app=is_app,
            state_factory=state_factory,
            f_stop=f_stop,
            run=run,
            flwr_dir=flwr_dir,
        )

    except Exception as ex:
        log(ERROR, "An exception occurred !! %s", ex)
        log(ERROR, traceback.format_exc())
        success = False
        raise RuntimeError("An error was encountered. Ending simulation.") from ex

    finally:
        # Trigger stop event
        f_stop.set()
        event(exit_event, event_details={"success": success})
        if serverapp_th:
            serverapp_th.join()
            if server_app_thread_has_exception.is_set():
                raise RuntimeError("Exception in ServerApp thread")

    log(DEBUG, "Stopping Simulation Engine now.")


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def _run_simulation(
    num_supernodes: int,
    exit_event: EventType,
    client_app: Optional[ClientApp] = None,
    server_app: Optional[ServerApp] = None,
    backend_name: str = "ray",
    backend_config: Optional[BackendConfig] = None,
    client_app_attr: Optional[str] = None,
    server_app_attr: Optional[str] = None,
    server_app_run_config: Optional[UserConfig] = None,
    app_dir: str = "",
    flwr_dir: Optional[str] = None,
    run: Optional[Run] = None,
    enable_tf_gpu_growth: bool = False,
    delay_start: int = 5,
    verbose_logging: bool = False,
    is_app: bool = False,
) -> None:
    """Launch the Simulation Engine."""
    if backend_config is None:
        backend_config = {}

    if "init_args" not in backend_config:
        backend_config["init_args"] = {}

    # Set default client_resources if not passed
    if "client_resources" not in backend_config:
        backend_config["client_resources"] = {"num_cpus": 2, "num_gpus": 0}

    # Initialization of backend config to enable GPU growth globally when set
    if "actor" not in backend_config:
        backend_config["actor"] = {"tensorflow": 0}

    # Set logging level
    logger = logging.getLogger("flwr")
    if verbose_logging:
        update_console_handler(level=DEBUG, timestamps=True, colored=True)
    else:
        backend_config["init_args"]["logging_level"] = backend_config["init_args"].get(
            "logging_level", WARNING
        )
        backend_config["init_args"]["log_to_driver"] = backend_config["init_args"].get(
            "log_to_driver", True
        )

    if enable_tf_gpu_growth:
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config.get("actor", {}).get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config["actor"]["tensorflow"] = True

    # Convert config to original JSON-stream format
    backend_config_stream = json.dumps(backend_config)

    # If no `Run` object is set, create one
    if run is None:
        run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
        run = Run.create_empty(run_id=run_id)

    args = (
        num_supernodes,
        backend_name,
        backend_config_stream,
        app_dir,
        is_app,
        enable_tf_gpu_growth,
        run,
        exit_event,
        delay_start,
        flwr_dir,
        client_app,
        client_app_attr,
        server_app,
        server_app_attr,
        server_app_run_config,
    )
    # Detect if there is an Asyncio event loop already running.
    # If yes, disable logger propagation. In environmnets
    # like Jupyter/Colab notebooks, it's often better to do this.
    asyncio_loop_running = False
    try:
        _ = (
            asyncio.get_running_loop()
        )  # Raises RuntimeError if no event loop is present
        log(DEBUG, "Asyncio event loop already running.")

        asyncio_loop_running = True

    except RuntimeError:
        pass

    finally:
        if asyncio_loop_running:
            # Set logger propagation to False to prevent duplicated log output in Colab.
            logger = set_logger_propagation(logger, False)

        _main_loop(*args)


def _parse_args_run_simulation() -> argparse.ArgumentParser:
    """Parse flower-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower simulation",
    )
    parser.add_argument(
        "--app",
        type=str,
        required=True,
        help="Path to a directory containing a FAB-like structure with a "
        "pyproject.toml.",
    )
    parser.add_argument(
        "--num-supernodes",
        type=int,
        required=True,
        help="Number of simulated SuperNodes.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Override configuration key-value pairs.",
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
        default="{}",
        help='A JSON formatted stream, e.g \'{"<keyA>":<value>, "<keyB>":<value>}\' to '
        "configure a backend. Values supported in <value> are those included by "
        "`flwr.common.typing.ConfigsRecordValues`. ",
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
        "--delay-start",
        type=int,
        default=3,
        help="Buffer time (in seconds) to delay the start the simulation engine after "
        "the `ServerApp`, which runs in a separate thread, has been launched.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="When unset, only INFO, WARNING and ERROR log messages will be shown. "
        "If set, DEBUG-level logs will be displayed. ",
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
        "--run-id",
        type=int,
        help="Sets the ID of the run started by the Simulation Engine.",
    )

    return parser
