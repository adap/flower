# Copyright 2023 Adap GmbH. All Rights Reserved.
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
"""Flower Telemetry."""

import datetime
import json
import os
import platform
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto

FLWR_TELEMETRY_ENABLED = os.getenv("FLWR_TELEMETRY_ENABLED", "1")
FLWR_TELEMETRY_LOGGING = os.getenv("FLWR_TELEMETRY_LOGGING", "0")

TELEMETRY_EVENTS_URL = "https://telemetry.flower.dev/api/v1/event"

# Using str as first base type to make it JSON serializable as
# otherwise the following exception will be thrown when serializing
# the event dict:
# TypeError: Object of type EventType is not JSON serializable
class EventType(str, Enum):
    """Types of telemetry events."""

    # This method combined with auto() will set the property value to
    # the property name e.g.
    # `START_CLIENT = auto()` becomes `START_CLIENT = "START_CLIENT"`
    # The type signature is not compatible with mypy, pylint and flake8
    # so each of those needs to be disabled for this line
    def _generate_next_value_(name, start, count, last_values):  # type: ignore # pylint: disable=no-self-argument,arguments-differ # noqa: E501
        return name

    # Client
    START_CLIENT = auto()
    STOP_CLIENT = auto()

    # Server
    START_SERVER = auto()
    STOP_SERVER = auto()

    # New Server
    RUN_SERVER = auto()
    END_SERVER = auto()

    # Simulation
    START_SIMULATION = auto()
    STOP_SIMULATION = auto()


# Use the ThreadPoolExecutor with max_workers=1 to have a queue
# as well as ensure that telemetry is unblocking.
executor = ThreadPoolExecutor(max_workers=1)


def event(event_type: EventType) -> Future[str]:
    """Submit create_event to ThreadPoolExecutor to avoid blocking."""
    return executor.submit(create_event, event_type)


def create_event(event_type: EventType) -> str:
    """Create telemetry event."""
    try:
        date = datetime.datetime.now().isoformat()
        context = {
            "date": date,
            "cpu": os.cpu_count(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "platform": platform.platform(),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "machine": platform.machine(),
                "architecture": platform.architecture(),
            },
        }

        data = {
            "event_type": event_type,
            "context": context,
        }

        data_json = json.dumps(data)

        if FLWR_TELEMETRY_LOGGING == "1":
            msg = " - ".join([date, "POST", data_json])
            print(msg)

        # If telemetry is not disabled with setting FLWR_TELEMETRY_ENABLED=0
        # create a request and send it to the telemetry backend
        if FLWR_TELEMETRY_ENABLED == "1":
            request = urllib.request.Request(
                url=TELEMETRY_EVENTS_URL,
                data=data_json.encode("utf-8"),
                headers={
                    "User-Agent": "flwr/123",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                result = response.read()

            response_json = str(json.loads(result.decode("utf-8")))

            return response_json  # The return value is mostly used for testing
    except Exception as ex:  # pylint: disable=broad-except
        print(ex)

    return ""  # In case telemetry is disabled
