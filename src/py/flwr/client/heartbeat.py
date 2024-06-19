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
"""Heartbeat utility functions."""


import threading
from typing import Callable

import grpc

from flwr.common.constant import PING_CALL_TIMEOUT
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential


def _ping_loop(ping_fn: Callable[[], None], stop_event: threading.Event) -> None:
    def wait_fn(wait_time: float) -> None:
        if not stop_event.is_set():
            stop_event.wait(wait_time)

    def on_backoff(state: RetryState) -> None:
        err = state.exception
        if not isinstance(err, grpc.RpcError):
            return
        status_code = err.code()
        # If ping call timeout is triggered
        if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            # Avoid long wait time.
            if state.actual_wait is None:
                return
            state.actual_wait = max(state.actual_wait - PING_CALL_TIMEOUT, 0.0)

    def wrapped_ping() -> None:
        if not stop_event.is_set():
            ping_fn()

    retrier = RetryInvoker(
        exponential,
        grpc.RpcError,
        max_tries=None,
        max_time=None,
        on_backoff=on_backoff,
        wait_function=wait_fn,
    )
    while not stop_event.is_set():
        retrier.invoke(wrapped_ping)


def start_ping_loop(
    ping_fn: Callable[[], None], stop_event: threading.Event
) -> threading.Thread:
    """Start a ping loop in a separate thread.

    This function initializes a new thread that runs a ping loop, allowing for
    asynchronous ping operations. The loop can be terminated through the provided stop
    event.
    """
    thread = threading.Thread(
        target=_ping_loop, args=(ping_fn, stop_event), daemon=True
    )
    thread.start()

    return thread