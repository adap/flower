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
APP_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"
PLATFORM_API_URL = "https://api.flower.ai/v1"

# Constants for federations
NOOP_FEDERATION = "default"

# Constants for exit handling
FORCE_EXIT_TIMEOUT_SECONDS = 5  # Used in `flwr_exit` function


class NodeStatus:
    """Event log writer types."""

    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"
    UNREGISTERED = "unregistered"

    def __new__(cls) -> NodeStatus:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
