# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower constants."""


from __future__ import annotations

MISSING_EXTRA_REST = """
Extra dependencies required for using the REST-based Fleet API are missing.

To use the REST API, install `flwr` with the `rest` extra:

    `pip install flwr[rest]`.
"""

TRANSPORT_TYPE_GRPC_BIDI = "grpc-bidi"
TRANSPORT_TYPE_GRPC_RERE = "grpc-rere"
TRANSPORT_TYPE_REST = "rest"
TRANSPORT_TYPE_VCE = "vce"
TRANSPORT_TYPES = [
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPE_VCE,
]

SUPEREXEC_DEFAULT_ADDRESS = "0.0.0.0:9093"

# Constants for ping
PING_DEFAULT_INTERVAL = 30
PING_CALL_TIMEOUT = 5
PING_BASE_MULTIPLIER = 0.8
PING_RANDOM_RANGE = (-0.1, 0.1)
PING_MAX_INTERVAL = 1e300

# Constants for FAB
APP_DIR = "apps"
FAB_CONFIG_FILE = "pyproject.toml"
FLWR_HOME = "FLWR_HOME"


GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY = "flower-version"
GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY = "should-exit"


class MessageType:
    """Message type."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    QUERY = "query"

    def __new__(cls) -> MessageType:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class MessageTypeLegacy:
    """Legacy message type."""

    GET_PROPERTIES = "get_properties"
    GET_PARAMETERS = "get_parameters"

    def __new__(cls) -> MessageTypeLegacy:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class SType:
    """Serialisation type."""

    NUMPY = "numpy.ndarray"

    def __new__(cls) -> SType:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class ErrorCode:
    """Error codes for Message's Error."""

    UNKNOWN = 0
    LOAD_CLIENT_APP_EXCEPTION = 1
    CLIENT_APP_RAISED_EXCEPTION = 2
    NODE_UNAVAILABLE = 3

    def __new__(cls) -> ErrorCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
