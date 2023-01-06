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
import uuid
from enum import Enum, auto

FLWR_TELEMETRY_ENABLED = os.getenv("FLWR_TELEMETRY_ENABLED", "1")
FLWR_TELEMETRY_LOGGING = os.getenv("FLWR_TELEMETRY_LOGGING", "0")

TELEMETRY_EVENTS_URL = "https://telemetry.flower.dev/api/v1/event"

# A random session to identify, for example, on which platforms a START
# event is emitted without a STOP event.
SESSION = str(uuid.uuid4())

# Using str as first base type to make it JSON serializable
# otherwise we will get the following exception:
# TypeError: Object of type EventType is not JSON serializable
class EventType(str, Enum):
    """Types of telemetry events."""

    # Checkout Python docs for why this code is here:
    # https://docs.python.org/3.7/library/enum.html#using-automatic-values
    # Also this function signature is super weird the only reasonable option
    # was to disable all checks for it.
    def _generate_next_value_(name, start, count, last_values):  # type: ignore # pylint: disable=no-self-argument,arguments-differ
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


class Colors:
    """Colors used for log messages."""

    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    ENDC = "\033[0m"


def send(
    event_type: EventType,
) -> str:
    """Send telemetry event."""
    try:
        date = datetime.datetime.now().isoformat()
        context = {
            "session": SESSION,
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

        event = {
            "event_type": event_type,
            "context": context,
        }

        event_json = json.dumps(event)

        if FLWR_TELEMETRY_LOGGING == "1":
            msg = " - ".join(
                [
                    f"{Colors.BLUE}{date}{Colors.ENDC}",
                    f"{Colors.CYAN}POST{Colors.ENDC}",
                    f"{Colors.GREEN}{event_json}{Colors.ENDC}",
                ]
            )
            print(msg)

        # If telemetry is not disabled with setting FLWR_TELEMETRY_ENABLED=0
        # create a request and send it to the telemetry backend
        if FLWR_TELEMETRY_ENABLED == "1":
            request = urllib.request.Request(
                url=TELEMETRY_EVENTS_URL,
                data=event_json.encode("utf-8"),
                headers={
                    "User-Agent": "flwr/123",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                response_json = str(json.loads(response.read().decode("utf-8")))

            return response_json  # The return value is mostly used for testing
    except Exception as ex:  # pylint: disable=broad-except
        print(ex)

    return ""  # In case telemetry is disabled
