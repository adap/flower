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


def _ping_loop(ping_fn: Callable[[], None], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            ping_fn()
        except grpc.RpcError:
            pass


def start_ping_loop(
    ping_fn: Callable[[], None], stop_event: threading.Event
) -> threading.Thread:
    """Start a ping loop in a separate thread.

    This function initializes a new thread that runs a ping loop, allowing for
    asynchronous ping operations. The loop can be terminated through the provided stop
    event.
    """
    thread = threading.Thread(target=_ping_loop, args=(ping_fn, stop_event))
    thread.start()

    return thread
