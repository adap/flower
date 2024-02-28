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


import sys
from signal import SIGINT, SIGTERM, signal
from threading import Thread
from types import FrameType
from typing import List, Optional

from grpc import Server

from flwr.common.telemetry import EventType, event


def register_exit_handlers(
    event_type: EventType,
    grpc_servers: Optional[List[Server]] = None,
    bckg_threads: Optional[List[Thread]] = None,
) -> None:
    """Register exit handlers for `SIGINT` and `SIGTERM` signals.

    Parameters
    ----------
    event_type : EventType
        The telemetry event that should be logged before exit.
    grpc_servers: Optional[List[Server]] (default: None)
        An otpional list of gRPC servers that need to be gracefully
        terminated before exiting.
    bckg_threads: Optional[List[Thread]] (default: None)
        An optional list of threads that need to be gracefully
        terminated before exiting.
    """
    default_handlers = {
        SIGINT: None,
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

        event_res = event(event_type=event_type)

        if grpc_servers is not None:
            for grpc_server in grpc_servers:
                grpc_server.stop(grace=1)

        if bckg_threads is not None:
            for bckg_thread in bckg_threads:
                bckg_thread.join()

        # Ensure event has happend
        event_res.result()

        # Setup things for graceful exit
        sys.exit(0)

    default_handlers[SIGINT] = signal(  # type: ignore
        SIGINT,
        graceful_exit_handler,  # type: ignore
    )
    default_handlers[SIGTERM] = signal(  # type: ignore
        SIGTERM,
        graceful_exit_handler,  # type: ignore
    )
