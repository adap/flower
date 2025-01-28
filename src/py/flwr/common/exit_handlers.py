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
"""Common function to register exit handlers for server and client."""


from signal import SIGINT, SIGQUIT, SIGTERM, signal
from threading import Thread
from types import FrameType
from typing import Optional

from grpc import Server

from flwr.common.telemetry import EventType

from .exit import ExitCode, flwr_exit

SIGNAL_TO_EXIT_CODE = {
    SIGINT: ExitCode.GRACEFUL_EXIT_SIGINT,
    SIGQUIT: ExitCode.GRACEFUL_EXIT_SIGQUIT,
    SIGTERM: ExitCode.GRACEFUL_EXIT_SIGTERM,
}


def register_exit_handlers(
    event_type: EventType,
    exit_message: Optional[str] = None,
    grpc_servers: Optional[list[Server]] = None,
    bckg_threads: Optional[list[Thread]] = None,
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
    """
    default_handlers = {
        SIGINT: None,
        SIGQUIT: None,
        SIGTERM: None,
    }

    def graceful_exit_handler(  # type: ignore
        signalnum,
        frame: FrameType,  # pylint: disable=unused-argument
    ) -> None:
        """Exit handler to be registered with `signal.signal`.

        When called will reset signal handler to original signal handler from
        default_handlers.
        """
        # Reset to default handler
        signal(signalnum, default_handlers[signalnum])

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

    default_handlers[SIGINT] = signal(  # type: ignore
        SIGINT,
        graceful_exit_handler,  # type: ignore
    )
    default_handlers[SIGQUIT] = signal(  # type: ignore
        SIGQUIT,
        graceful_exit_handler,  # type: ignore
    )
    default_handlers[SIGTERM] = signal(  # type: ignore
        SIGTERM,
        graceful_exit_handler,  # type: ignore
    )
