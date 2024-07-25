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
"""Simulation engine executor."""


import subprocess
import sys
from logging import ERROR, INFO, WARN
from typing import Optional

from typing_extensions import override

from flwr.cli.config_utils import load_and_validate
from flwr.cli.install import install_from_fab
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.superlink.state.utils import generate_rand_int_from_bytes

from .executor import Executor, RunTracker


def _user_config_to_str(user_config: UserConfig) -> str:
    """Convert override user config to string."""
    user_config_list_str = []
    for key, value in user_config.items():
        if isinstance(value, bool):
            user_config_list_str.append(f"{key}={str(value).lower()}")
        elif isinstance(value, (int, float)):
            user_config_list_str.append(f"{key}={value}")
        elif isinstance(value, str):
            user_config_list_str.append(f'{key}="{value}"')
        else:
            raise ValueError(
                "Only types `bool`, `float`, `int` and `str` are supported"
            )

    user_config_str = ",".join(user_config_list_str)
    return user_config_str


class SimulationEngine(Executor):
    """Simulation engine executor.

    Parameters
    ----------
    num_supernodes: Opitonal[str] (default: None)
        Total number of nodes to involve in the simulation.
    """

    def __init__(
        self,
        num_supernodes: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.num_supernodes = num_supernodes
        self.verbose = verbose

    @override
    def set_config(
        self,
        config: UserConfig,
    ) -> None:
        """Set executor config arguments.

        Parameters
        ----------
        config : UserConfig
            A dictionary for configuration values.
            Supported configuration key/value pairs:
            - "num-supernodes": int
                Number of nodes to register for the simulation.
            - "verbose": bool
                Set verbosity of logs.
        """
        if num_supernodes := config.get("num-supernodes"):
            if not isinstance(num_supernodes, int):
                raise ValueError("The `num-supernodes` value should be of type `int`.")
            self.num_supernodes = num_supernodes
        elif self.num_supernodes is None:
            log(
                ERROR,
                "To start a run with the simulation plugin, please specify "
                "the number of SuperNodes. This can be done by using the "
                "`--executor-config` argument when launching the SuperExec.",
            )
            raise ValueError(
                "`num-supernodes` must not be `None`, it must be a valid "
                "positive integer."
            )

        if verbose := config.get("verbose"):
            self.verbose = verbose

    @override
    def start_run(
        self,
        fab_file: bytes,
        override_config: UserConfig,
        federation_config: UserConfig,
    ) -> Optional[RunTracker]:
        """Start run using the Flower Simulation Engine."""
        if self.num_supernodes is None:
            raise ValueError(
                "Error in `SuperExec` (`SimulationEngine` executor):\n\n"
                "`num-supernodes` must not be `None`, it must be a valid "
                "positive integer. In order to start this simulation executor "
                "with a specified number of `SuperNodes`, you can either provide "
                "a `--executor` that has been initialized with a number of nodes "
                "to the `flower-superexec` CLI, or `--executor-config num-supernodes=N`"
                "to the `flower-superexec` CLI."
            )
        try:

            # Install FAB to flwr dir
            fab_path = install_from_fab(fab_file, None, True)

            # Prepare FAB install command
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                str(fab_path),
            ]
            # Install FAB Python package
            if self.verbose:
                subprocess.run(command, check=True)
            else:
                subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

            # Load and validate config
            config, errors, warnings = load_and_validate(fab_path / "pyproject.toml")
            if errors:
                raise ValueError(errors)

            if warnings:
                log(WARN, warnings)

            if config is None:
                raise ValueError(
                    "Config extracted from FAB's pyproject.toml is not valid"
                )

            # In Simulation there is no SuperLink, still we create a run_id
            run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
            log(INFO, "Created run %s", str(run_id))

            # Prepare commnand
            command = [
                "flower-simulation",
                "--app",
                f"{str(fab_path)}",
                "--num-supernodes",
                f"{federation_config.get('num-supernodes', self.num_supernodes)}",
                "--run-id",
                str(run_id),
            ]

            if override_config:
                override_config_str = _user_config_to_str(override_config)
                command.extend(["--run-config", f"{override_config_str}"])

            # Start Simulation
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                text=True,
            )

            log(INFO, "Started run %s", str(run_id))

            return RunTracker(
                run_id=run_id,
                proc=proc,
            )

        # pylint: disable-next=broad-except
        except Exception as e:
            log(ERROR, "Could not start run: %s", str(e))
            return None


executor = SimulationEngine()
