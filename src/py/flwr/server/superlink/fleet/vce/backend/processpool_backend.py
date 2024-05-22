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
"""ProcessPool Backend for Fleet API using the Simulation Engine."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from logging import DEBUG
from multiprocessing import Manager, Queue
from pathlib import Path
from queue import Empty
from threading import Event
from typing import Callable, Tuple

import cloudpickle

from flwr.client.client_app import ClientApp
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message

from .backend import Backend, BackendConfig


def run_client_app(
    in_queue: "Queue[Tuple[bytes,Message,Context]]",
    out_queue: "Queue[Tuple[Message,Context]]",
    f_stop: Event,
) -> None:
    """Run a ClientApp processing a Message."""
    while not f_stop.is_set():
        try:
            client_app_pkl, message, context = in_queue.get(timeout=1)
        except Empty:
            # only break if f_stop is set
            pass
        else:
            client_app = pickle.loads(client_app_pkl)
            out_message = client_app(message=message, context=context)
            out_queue.put((out_message, context))


class ProcessPoolBackend(Backend):
    """A backend that submits job to a ProcessPoolExecutor."""

    def __init__(
        self,
        backend_config: BackendConfig,
        work_dir: str,
    ) -> None:
        """."""
        log(DEBUG, "Initialising: %s", self.__class__.__name__)
        log(DEBUG, "Backend config: %s", backend_config)

        if not Path(work_dir).exists():
            raise ValueError(f"Specified work_dir {work_dir} does not exist.")

        self.manager = Manager()
        self.f_stop = self.manager.Event()
        self.in_queue = self.manager.Queue()
        self.out_queue = self.manager.Queue()

        # TODO: read from backend_config
        self.num_proc = 8

        self.executor = ProcessPoolExecutor(self.num_proc)

    @property
    def num_workers(self) -> int:
        """."""
        return self.num_proc

    def is_worker_idle(self) -> bool:
        """."""
        return False

    def build(self) -> None:
        """."""
        for _ in range(self.num_proc):
            self.executor.submit(
                run_client_app,
                self.in_queue,  # type: ignore
                self.out_queue,  # type: ignore
                self.f_stop,
            )

        log(DEBUG, "Constructed ProcessPoolExecutor with: %i processes", self.num_proc)

    def process_message(
        self, app: Callable[[], ClientApp], message: Message, context: Context
    ) -> Tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        # put stuff in queue
        client_app = app()
        self.in_queue.put((cloudpickle.dumps(client_app), message, context))

        # wait for response
        out_message, updated_context = self.out_queue.get()
        return out_message, updated_context

    def terminate(self) -> None:
        """Terminate backend shutting down the Manager and Executor."""
        self.manager.shutdown()
        self.executor.shutdown()
