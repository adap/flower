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
"""ThreadPool backend for the Fleet API using the Simulation Engine."""

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from logging import DEBUG, ERROR, INFO
from typing import Callable, Tuple

from flwr.client.client_app import ClientApp
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message

from .backend import Backend, BackendConfig


def _run_client_app(
    client_app_fn: Callable[[], ClientApp],
    message: Message,
    context: Context,
) -> Tuple[Message, Context]:

    # Load app
    app: ClientApp = client_app_fn()

    # Handle task message
    out_message = app(message=message, context=context)

    return out_message, context


class ThreadPoolExecutorBackend(Backend):

    def __init__(self, backend_config: BackendConfig, work_dir: str) -> None:
        """Prepare ThreadPoolBackend."""
        log(INFO, "Initialising: %s", self.__class__.__name__)
        log(INFO, "Backend config: %s", backend_config)

        # If not set, it will use default: min(32, os.cpu_count() + 4)
        # as per https://docs.python.org/3/library/concurrent.futures.html
        self.max_workers = backend_config.get("max_workers", None)

        self.pool = None

        # Maing event loop
        self.loop = asyncio.get_running_loop()

    @property
    def num_workers(self) -> int:
        return self.pool._max_workers

    async def build(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        log(
            DEBUG,
            "Built %s with %i max_workers",
            self.pool.__class__.__name__,
            self.pool._max_workers,
        )

    def is_worker_idle(self) -> bool:
        raise NotImplementedError()

    async def process_message(
        self, app: Callable[[], ClientApp], message: Message, context: Context
    ) -> Tuple[Message, Context]:

        try:
            future = self.loop.run_in_executor(
                self.pool, _run_client_app, app, message, context
            )

            # Await result
            out_mssg, updated_context = await future

            return out_mssg, updated_context

        except Exception as ex:
            log(
                ERROR,
                "An exception was raised when processing a message by %s",
                self.__class__.__name__,
            )
            log(ERROR, traceback.format_exc())
            raise ex

    async def terminate(self) -> None:
        self.pool.shutdown(cancel_futures=True)
