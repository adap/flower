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
"""Constants for Flower infrastructure."""


from __future__ import annotations

# Top-level key in YAML config for exec plugin settings
EXEC_PLUGIN_SECTION = "exec_plugin"

# Flower in-memory Python-based database name
FLWR_IN_MEMORY_DB_NAME = ":flwr-in-memory:"

# Constants for Hub
APP_ID_PATTERN = r"^@(?P<user>[^/]+)/(?P<app>[^/]+)$"
PLATFORM_API_URL = "https://api.flower.ai/v1"

# App spec
ALLOWED_EXTS = [".py", ".toml", ".md"]
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_FILE_COUNT = 1000
MAX_DIR_DEPTH = 10  # relative depth (number of parts in relpath)
UTF8 = "utf-8"
MIME_MAP = {
    ".py": "text/x-python; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".toml": "application/toml; charset=utf-8",
}

# Constants for federations
NOOP_FEDERATION = "default"


class NodeStatus:
    """Event log writer types."""

    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"
    UNREGISTERED = "unregistered"

    def __new__(cls) -> NodeStatus:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
