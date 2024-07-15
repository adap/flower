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


class SimulationEngine(Executor):
    """Simulation engine executor.

    Parameters
    ----------
    num_supernodes: Opitonal[str] (default: None)
        Total number of nodes to involve in the simulation.
    """

    def __init__(
        self,
        num_supernodes: Optional[str] = None,
    ) -> None:
        self.num_supernodes = num_supernodes

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
            - "num-supernodes": str
                Number of nodes to register for the simulation.
        """
        if not config:
            return
        if num_supernodes := config.get("num-supernodes"):
            self.num_supernodes = num_supernodes

        # Validate config
        if self.num_supernodes is None:
            log(
                ERROR,
                "To start a run with the simulation plugin, please specify "
                "the number of SuperNodes. This can be done by using the "
                "`--executor-config` argument when launching the SuperExec.",
            )
            raise ValueError("`num-supernodes` must not be `None`")

    @override
    def start_run(
        self, fab_file: bytes, override_config: UserConfig
    ) -> Optional[RunTracker]:
        """Start run using the Flower Simulation Engine."""
        try:
            if override_config:
                raise ValueError(
                    "Overriding the run config is not yet supported with the "
                    "simulation executor.",
                )

            # Install FAB to flwr dir
            fab_path = install_from_fab(fab_file, None, True)

            # Install FAB Python package
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-deps", str(fab_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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

            # Get ClientApp and SeverApp components
            flower_components = config["flower"]["components"]
            clientapp = flower_components["clientapp"]
            serverapp = flower_components["serverapp"]

            # In Simulation there is no SuperLink, still we create a run_id
            run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
            log(INFO, "Created run %s", str(run_id))

            # Prepare commnand
            command = [
                "flower-simulation",
                "--client-app",
                f"{clientapp}",
                "--server-app",
                f"{serverapp}",
                "--num-supernodes",
                f"{self.num_supernodes}",
                "--run-id",
                str(run_id),
            ]

            # Start Simulation
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
