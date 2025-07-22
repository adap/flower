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
"""`flwr-app-scheduler` command."""


import argparse
from logging import INFO

from flwr.common import EventType, event
from flwr.common.logger import log


def flwr_app_scheduler() -> None:
    """Run `flwr-app-scheduler` command."""
    args = _parse_args().parse_args()

    # Log the first message after parsing arguments in case of `--help`
    log(INFO, "Starting Flower App Scheduler")

    # Trigger telemetry event
    event(EventType.FLWR_APP_SCHEDULER_RUN_ENTER)

    raise NotImplementedError(
        "The `flwr-app-scheduler` command is not implemented yet."
    )


def _parse_args() -> argparse.ArgumentParser:
    """Parse `flwr-app-scheduler` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower App Scheduler",
    )
    parser.add_argument(
        "--appio-api-address", type=str, required=True, help="Address of the AppIO API"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Connect to the AppIO API without TLS. "
        "Data transmitted between the client and server is not encrypted. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--flwr-dir",
        default=None,
        help="""The path containing installed Flower Apps.
        By default, this value is equal to:

            - `$FLWR_HOME/` if `$FLWR_HOME` is defined
            - `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined
            - `$HOME/.flwr/` in all other cases
        """,
    )
    return parser
