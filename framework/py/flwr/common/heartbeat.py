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
"""Heartbeat sender."""


import random
import threading
from typing import Callable

from .constant import (
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from .retry_invoker import RetryInvoker, exponential


class PingFailure(Exception):
    """Exception raised when ping fails."""


class HeartbeatSender:
    """Send periodic heartbeat pings to a server in a background thread.

    This class uses a user-provided `ping_fn` to send heartbeats. If a ping fails,
    it is retried using an exponential backoff strategy.

    Parameters
    ----------
    ping_fn : Callable[[], bool]
        Function used to send a ping. Should return True if the ping succeeds,
        or False if it fails. Any internal exceptions (e.g., gRPC errors)
        should be handled within this function.
    """

    def __init__(
        self,
        ping_fn: Callable[[], bool],
    ) -> None:
        self.ping_fn = ping_fn
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._retry_invoker = RetryInvoker(
            lambda: exponential(max_delay=20),
            PingFailure,  # The only exception we want to retry on
            max_tries=None,
            max_time=None,
            # Allow the stop event to interrupt the wait
            wait_function=self._stop_event.wait,  # type: ignore
        )

    def start(self) -> None:
        """Start the heartbeat sender."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat sender."""
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        """Run the heartbeat sender."""
        while not self._stop_event.is_set():
            # Ping the server and retry if it fails
            self._retry_invoker.invoke(self._ping)

            # Calculate the interval for the next ping
            # Formula: next_interval = (interval - timeout) * random.uniform(0.7, 0.9)
            rd = random.uniform(*PING_RANDOM_RANGE)
            next_interval: float = PING_DEFAULT_INTERVAL - PING_CALL_TIMEOUT
            next_interval *= PING_BASE_MULTIPLIER + rd

            # Wait for the next ping
            self._stop_event.wait(next_interval)

    def _ping(self) -> None:
        """Ping the server and raise an exception if it fails."""
        # Check if the stop event is set before sending a ping
        if not self._stop_event.is_set():
            # Trigger retry if ping fails
            if not self.ping_fn():
                raise PingFailure
