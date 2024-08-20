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
from typing import Any, Dict, List, Optional

from flwr.common import Message
from flwr.common.typing import Run


class SuperNodeTracker:
    """A utility class for tracking and recording SuperNode."""

    def __init__(self, supernode_id: Optional[int]) -> None:
        self.supernode_id = str(supernode_id)
        self.filename = f"supernode_tracking_({supernode_id}).json"
        self.run_ids: List[int] = []

        # Create an empty file if it does not exist
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8"):
                pass

    def record_run(self, run: Run) -> None:
        """Record the run metadata."""
        timestamp = datetime.now().timestamp()
        run_metadata = {
            "run_id": run.run_id,
            "fab_id": run.fab_id,
            "fab_version": run.fab_version,
        }

        record = {"timestamp": timestamp, "run": {"metadata": run_metadata}}

        if run.run_id not in self.run_ids:
            self.save_to_file(record)
            self.run_ids.append(run.run_id)

    def record_message(
        self, from_entity: str, to_entity: str, message: Message
    ) -> None:
        """Record the message metadata."""
        timestamp = datetime.now().timestamp()

        message_size = None
        if message.has_content():
            message_size = self.compute_message_size(message)

        message_error = None
        if message.has_error():
            message_error = message.__dict__["_error"]

        record = {
            "timestamp": timestamp,
            "from": from_entity,
            "to": to_entity,
            "message": {
                "metadata": message.metadata.__dict__,
                "message_size": message_size,
                "error": message_error,
            },
        }
        self.save_to_file(record)

    def save_to_file(self, data: Dict[str, Any]) -> None:
        """Write data to the JSON file."""
        try:
            with open(self.filename, "a", encoding="utf-8") as file:
                json.dump(data, file)
                file.write("\n")
        except OSError as e:
            raise RuntimeError(f"Failed to write to file at {self.filename}") from e

    def compute_message_size(self, message: Message) -> int:
        """Compute the message size in bytes."""
        message_size_in_bytes = 0

        for p_record in message.content.parameters_records.values():
            message_size_in_bytes += p_record.count_bytes()

        for c_record in message.content.configs_records.values():
            message_size_in_bytes += c_record.count_bytes()

        for m_record in message.content.metrics_records.values():
            message_size_in_bytes += m_record.count_bytes()

        return message_size_in_bytes
