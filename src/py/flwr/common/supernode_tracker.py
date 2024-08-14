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
from typing import Any, Dict


class SuperNodeTracker:
    """A utility class for tracking and recording SuperNode."""

    file_path = ""
    records_holder = []

    @staticmethod
    def create_tracking_file(file_path: str) -> None:
        """Initialize the static tracker with a file path."""
        SuperNodeTracker.file_path = file_path
        # Create an empty file if it doesn't exist
        if not os.path.exists(SuperNodeTracker.file_path):
            open(SuperNodeTracker.file_path, "w").close()

    @staticmethod
    def record_message_metadata(entity: str, metadata: Dict[str, Any]) -> None:
        """Add a log entry to the records holder."""
        if metadata:
            log_entry = {entity: metadata}
            SuperNodeTracker.records_holder.append(log_entry)

    @staticmethod
    def save_to_file() -> None:
        """Write all accumulated record entries to the file."""
        with open(SuperNodeTracker.file_path, "a") as file:
            for entry in SuperNodeTracker.records_holder:
                json.dump(entry, file)
                file.write("\n")
        SuperNodeTracker.records_holder.clear()

    @staticmethod
    def clear_records_holder() -> None:
        """Clear the records holder without saving to file."""
        SuperNodeTracker.records_holder.clear()
