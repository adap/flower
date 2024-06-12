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
"""Simulation engine executor plugin."""


import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab

from .executor import Executor, Run


class SimulationEngine(Executor):
    """Simulation engine executor plugin."""

    def _install_fab(self, fab_file: bytes) -> Path:
        return install_from_fab(fab_file, None, True)

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> Run:
        """Start run using the Flower Simulation Engine."""
        _ = ttl
        _, fab_id, engine_conf = get_fab_metadata(fab_file)

        run_id = int.from_bytes(os.urandom(8), "little", signed=True)

        fab_path = self._install_fab(fab_file)

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(fab_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        fab_name = Path(fab_id).name

        num_nodes = engine_conf["simulation"]["supernode"]["num"]

        return Run(
            run_id=run_id,
            proc=subprocess.Popen(
                [
                    "flower-simulation",
                    "--client-app",
                    f"{fab_name}.client:app",
                    "--server-app",
                    f"{fab_name}.server:app",
                    "--num-supernodes",
                    f"{num_nodes}",
                    "--run-id",
                    str(run_id),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )


simulation_plugin = SimulationEngine()
