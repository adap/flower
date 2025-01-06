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
"""Run ServerApp."""


from logging import DEBUG, ERROR
from typing import Optional

from flwr.common import Context, EventType, event
from flwr.common.exit_handlers import register_exit_handlers
from flwr.common.logger import log
from flwr.common.object_ref import load_app

from .driver import Driver
from .server_app import LoadServerAppError, ServerApp


def run(
    driver: Driver,
    context: Context,
    server_app_dir: str,
    server_app_attr: Optional[str] = None,
    loaded_server_app: Optional[ServerApp] = None,
) -> Context:
    """Run ServerApp with a given Driver."""
    if not (server_app_attr is None) ^ (loaded_server_app is None):
        raise ValueError(
            "Either `server_app_attr` or `loaded_server_app` should be set "
            "but not both."
        )

    # Load ServerApp if needed
    def _load() -> ServerApp:
        if server_app_attr:
            server_app: ServerApp = load_app(
                server_app_attr, LoadServerAppError, server_app_dir
            )

            if not isinstance(server_app, ServerApp):
                raise LoadServerAppError(
                    f"Attribute {server_app_attr} is not of type {ServerApp}",
                ) from None

        if loaded_server_app:
            server_app = loaded_server_app
        return server_app

    server_app = _load()

    # Call ServerApp
    server_app(driver=driver, context=context)

    log(DEBUG, "ServerApp finished running.")
    return context


def run_server_app() -> None:
    """Run Flower server app."""
    event(EventType.RUN_SERVER_APP_ENTER)
    log(
        ERROR,
        "The command `flower-server-app` has been replaced by `flwr run`.",
    )
    register_exit_handlers(event_type=EventType.RUN_SERVER_APP_LEAVE)
