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
"""Test Fleet Simulation Engine API."""

import asyncio
import threading
from time import sleep
from unittest import IsolatedAsyncioTestCase

from flwr.server.superlink.state import StateFactory

from . import start_vce


class AsyncTestFleetSimulationEngine(IsolatedAsyncioTestCase):
    """A basic class to test Fleet Simulation Enginge funcionality."""

    def test_start_and_shutdown(self) -> None:
        """Start Simulation Engine Fleet and terminate it."""
        f_stop = asyncio.Event()

        # Initialize StateFactory
        state_factory = StateFactory(":flwr-in-memory-state:")

        superlink_th = threading.Thread(
            target=start_vce,
            args=(
                50,
                "",
                "ray",
                "{}",  # an empty json stream (represents an empty config)
                state_factory,
                "",
                f_stop,
            ),
            daemon=False,
        )

        superlink_th.start()

        # Sleep for some time
        sleep(10)

        # Trigger stop event
        f_stop.set()

        superlink_th.join()
