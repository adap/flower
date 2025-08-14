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
"""Common function to register exit handlers for server and client."""


import signal
from threading import Thread
from types import FrameType
from typing import Callable, Optional
import secrets

from grpc import Server
import sys
from flwr.common.telemetry import EventType
import faulthandler
import threading
from .exit import ExitCode, flwr_exit

faulthandler.enable()

SIGNAL_TO_EXIT_CODE: dict[int, int] = {
    signal.SIGINT: ExitCode.GRACEFUL_EXIT_SIGINT,
    signal.SIGTERM: ExitCode.GRACEFUL_EXIT_SIGTERM,
}
_lock_exit_handlers = threading.Lock()
registered_exit_handlers: dict[str, Callable[[], None]] = {}

# SIGQUIT is not available on Windows
if hasattr(signal, "SIGQUIT"):
    SIGNAL_TO_EXIT_CODE[signal.SIGQUIT] = ExitCode.GRACEFUL_EXIT_SIGQUIT


def register_exit_handlers(
    event_type: EventType,
    exit_message: Optional[str] = None,
    grpc_servers: Optional[list[Server]] = None,
    bckg_threads: Optional[list[Thread]] = None,
    exit_handlers: Optional[list[Callable[[], None]]] = None,
) -> None:
    """Register exit handlers for `SIGINT`, `SIGTERM` and `SIGQUIT` signals.

    Parameters
    ----------
    event_type : EventType
        The telemetry event that should be logged before exit.
    exit_message : Optional[str] (default: None)
        The message to be logged before exiting.
    grpc_servers: Optional[List[Server]] (default: None)
        An otpional list of gRPC servers that need to be gracefully
        terminated before exiting.
    bckg_threads: Optional[List[Thread]] (default: None)
        An optional list of threads that need to be gracefully
        terminated before exiting.
    exit_handlers: Optional[List[Callable[[], None]]] (default: None)
        An optional list of exit handlers to be called before exiting.
        Additional exit handlers can be added using `add_exit_handler`.
    """
    default_handlers: dict[int, Callable[[int, FrameType], None]] = {}
    for handler in exit_handlers or []:
        add_exit_handler(handler)

    def graceful_exit_handler(signalnum: int, _frame: FrameType) -> None:
        """Exit handler to be registered with `signal.signal`.

        When called will reset signal handler to original signal handler from
        default_handlers.
        """
        print(f"Received signal {signalnum}, exiting gracefully...")
        # Reset to default handler
        signal.signal(signalnum, default_handlers[signalnum])  # type: ignore

        with _lock_exit_handlers:
            for handler in registered_exit_handlers.values():
                handler()

        if grpc_servers is not None:
            for grpc_server in grpc_servers:
                grpc_server.stop(grace=1)

        if bckg_threads is not None:
            for bckg_thread in bckg_threads:
                bckg_thread.join()
        # Setup things for graceful exit
        flwr_exit(
            code=SIGNAL_TO_EXIT_CODE[signalnum],
            message=exit_message,
            event_type=event_type,
        )

    # Register signal handlers
    for sig in SIGNAL_TO_EXIT_CODE:
        default_handler = signal.signal(sig, graceful_exit_handler)  # type: ignore
        default_handlers[sig] = default_handler  # type: ignore


def add_exit_handler(exit_handler: Callable[[], None]) -> None:
    """Add an exit handler to be called on graceful exit.

    This function allows you to register additional exit handlers
    that will be executed when the application exits gracefully,
    if `register_exit_handlers` was called.

    Parameters
    ----------
    exit_handler : Callable[[], None]
        A callable that takes no arguments and performs cleanup or
        other actions before the application exits.

    Returns
    -------
    str
        A unique ID for the exit handler, which can be used to remove 
        it later using `remove_exit_handler`.

    Notes
    -----
    This method is not thread-safe, and it allows you to add the
    same exit handler multiple times.
    """
    with _lock_exit_handlers:
        handler_id = secrets.token_hex(32)
        registered_exit_handlers[handler_id] = exit_handler
        return handler_id


def remove_exit_handler(handler_id: str) -> None:
    """Remove an exit handler by its ID.

    Parameters
    ----------
    handler_id : str
        The ID of the exit handler to remove.
    """
    with _lock_exit_handlers:
        registered_exit_handlers.pop(handler_id, None)
