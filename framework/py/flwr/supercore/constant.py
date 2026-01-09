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

from flwr.common.constant import FLWR_DIR

# Top-level key in YAML config for exec plugin settings
EXEC_PLUGIN_SECTION = "exec_plugin"

# Flower in-memory Python-based database name
FLWR_IN_MEMORY_DB_NAME = ":flwr-in-memory:"

# Constants for Hub
APP_ID_PATTERN = r"^@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
APP_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"
PLATFORM_API_URL = "https://api.flower.ai/v1"

# Specification for app publishing
APP_PUBLISH_INCLUDE_PATTERNS = (
    "**/*.py",
    "**/*.toml",
    "**/*.md",
)
APP_PUBLISH_EXCLUDE_PATTERNS = FAB_EXCLUDE_PATTERNS = (
    f"{FLWR_DIR}/**",  # Exclude the .flwr directory
    "**/__pycache__/**",
)
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

# Constants for exit handling
FORCE_EXIT_TIMEOUT_SECONDS = 5  # Used in `flwr_exit` function

# Constants for message processing timing
MESSAGE_TIME_ENTRY_MAX_AGE_SECONDS = 3600

# System message type
SYSTEM_MESSAGE_TYPE = "system"


class NodeStatus:
    """Event log writer types."""

    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"
    UNREGISTERED = "unregistered"

    def __new__(cls) -> NodeStatus:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


# The default configuration for the global config file
DEFAULT_CONFIG_TOML = """[superlink]
default = "local-simulation"

[superlink.supergrid]
address = "supergrid.flower.ai"
enable-account-auth = true
federation = "<federation-name>"

[superlink.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0
"""
