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


import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab

from .executor import Executor, RunTracker


class SimulationEngine(Executor):
    """Simulation engine executor."""

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> RunTracker:
        """Start run using the Flower Simulation Engine."""
        _ = ttl
        _, fab_id = get_fab_metadata(fab_file)

        run_id = int.from_bytes(os.urandom(8), "little", signed=True)

        fab_path = install_from_fab(fab_file, None, True)

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(fab_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        fab_name = Path(fab_id).name

        # TODO: return engine_conf from `get_fab_metadata`
        # num_nodes = engine_conf["simulation"]["supernode"]["num"]
        num_nodes = 2

        return RunTracker(
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


executor = SimulationEngine()
