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
"""SuperNode Tracker."""

import json
import os
from datetime import datetime
from typing import Any, Dict

from flwr.common.typing import Run


class SuperNodeTracker:
    """A utility class for tracking and recording SuperNode."""

    def __init__(self, supernode_id: str) -> None:
        self.supernode_id = supernode_id
        self.filename = f"supernode_tracking_({supernode_id}).json"
        self.run_ids = []

        # Create an empty file if it does not exist
        if not os.path.exists(self.filename):
            open(self.filename, "w").close()

    def record_run_metadata(self, run: Run) -> None:
        """Set the run metadata."""
        run_metadata = {
            "run_id": run.run_id,
            "fab_id": run.fab_id,
            "fab_version": run.fab_version,
        }

        if run.run_id not in self.run_ids:
            self.save_to_file({"run": run_metadata})
            self.run_ids.append(run.run_id)

    def record_message_metadata(self, entity: str, metadata: Dict[str, Any]) -> None:
        """Record metadata with a timestamp."""
        timestamp = datetime.now().timestamp()
        record = {"timestamp": timestamp, entity: {"message_metadata": metadata}}
        self.save_to_file(record)

    def save_to_file(self, data: Dict[str, Any]) -> None:
        """Write data to the JSON file."""
        try:
            with open(self.filename, "a", encoding="utf-8") as file:
                json.dump(data, file)
                file.write("\n")
        except OSError as e:
            # Raise a new exception while preserving the original context
            raise RuntimeError(f"Failed to write to file at {self.filename}") from e
