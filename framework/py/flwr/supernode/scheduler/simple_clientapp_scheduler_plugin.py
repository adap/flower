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
"""Simple Flower ClientApp Scheduler plugin."""


import os
import subprocess
from collections.abc import Sequence
from typing import Optional

from flwr.supercore.scheduler import SchedulerPlugin


class SimpleClientAppSchedulerPlugin(SchedulerPlugin):
    """Simple Flower ClientApp Scheduler plugin.

    The plugin always selects the first candidate run ID.
    """

    def select_run_id(self, candidate_run_ids: Sequence[int]) -> Optional[int]:
        """Select a run ID to execute from a sequence of candidates."""
        if not candidate_run_ids:
            return None
        return candidate_run_ids[0]

    def launch_app(self, token: str, run_id: int) -> None:
        """Launch the application associated with a given run ID and token."""
        cmds = ["flwr-clientapp", "--insecure"]
        cmds += ["--clientappio-api-address", self.appio_api_address]
        cmds += ["--token", token]
        cmds += ["--parent-pid", str(os.getpid())]
        cmds += ["--flwr-dir", self.flwr_dir]
        # Launch the client app without waiting for it to complete.
        # Since we don't need to manage the process, we intentionally avoid using
        # a `with` statement. Suppress the pylint warning for it in this case.
        # pylint: disable-next=consider-using-with
        subprocess.Popen(cmds)
