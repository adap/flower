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
from typing import Any, Dict, List


class SuperNodeTracker:
    """A utility class for tracking and recording SuperNode."""

    def __init__(self, file_path: str) -> None:
        """Initialize the tracker with a file path."""
        self.file_path = file_path
        self.records_holder: List[Dict[str, Dict[str, Any]]] = []

        # Create an empty file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8"):
                pass

    def record_message_metadata(self, entity: str, metadata: Dict[str, Any]) -> None:
        """Add a log entry to the records holder."""
        log_entry = {entity: metadata}
        self.records_holder.append(log_entry)

    def save_to_file(self) -> None:
        """Write all accumulated record entries to the file."""
        with open(self.file_path, "a", encoding="utf-8") as file:
            for entry in self.records_holder:
                json.dump(entry, file)
                file.write("\n")
        self.records_holder.clear()

    def clear_records_holder(self) -> None:
        """Clear the records holder without saving to file."""
        self.records_holder.clear()
