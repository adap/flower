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

import time
from logging import DEBUG
from multiprocessing import Manager, Process, Queue
from queue import Empty
from threading import Event
from typing import List, Tuple

from flwr.client.supernode.app import _get_load_client_app_fn
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message

from .backend import Backend, BackendConfig


def run_client_app(
    client_app_attr,
    app_dir,
    flwr_dir,
    in_queue: "Queue[Tuple[bytes,Message,Context]]",
    out_queue: "Queue[Tuple[Message,Context]]",
    f_stop: Event,
) -> None:
    """Run a ClientApp processing a Message."""
    # print('hey')
    print("loading client app in process...")
    t_start = time.time()
    client_app = _get_load_client_app_fn(
        default_app_ref=client_app_attr,
        app_path=app_dir,
        flwr_dir=flwr_dir,
        multi_app=False,
    )("", "")
    print(f"Loaded (took: {time.time() - t_start}s)")
    while not f_stop.is_set():
        try:
            message, context = in_queue.get(timeout=1)
            # print('got message tot run')
        except Empty:
            # only break if f_stop is set
            pass
        else:
            out_message = client_app(message=message, context=context)
            out_queue.put((out_message, context))


class ProcessPoolBackend(Backend):
    """A backend that submits job to a ProcessPoolExecutor."""

    def __init__(
        self,
        backend_config: BackendConfig,
    ) -> None:
        """."""
        log(DEBUG, "Initialising: %s", self.__class__.__name__)
        log(DEBUG, "Backend config: %s", backend_config)

        self.manager = Manager()
        self.in_queue = self.manager.Queue()
        self.out_queue = self.manager.Queue()
        self.f_stop = self.manager.Event()

        # TODO: read from backend_config
        self.num_proc = 5

        # self.executor = ProcessPoolExecutor(self.num_proc)
        self.processes: List[Process] = []

    @property
    def num_workers(self) -> int:
        """."""
        return self.num_proc

    def is_worker_idle(self) -> bool:
        """."""
        return False

    def build(
        self,
        client_app_attr,
        app_dir,
        flwr_dir,
    ) -> None:
        """."""
        for _ in range(self.num_proc):
            p = Process(
                target=run_client_app,
                args=(
                    client_app_attr,
                    app_dir,
                    flwr_dir,
                    self.in_queue,
                    self.out_queue,
                    self.f_stop,
                ),
                daemon=True,
            )
            self.processes.append(p)
            p.start()

        log(DEBUG, "Constructed ProcessPoolExecutor with: %i processes", self.num_proc)

    def process_message(
        self, message: Message, context: Context
    ) -> Tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        # put stuff in queue
        self.in_queue.put((message, context))

        # wait for response
        out_message, updated_context = self.out_queue.get()
        return out_message, updated_context

    def terminate(self) -> None:
        """Terminate backend shutting down the Manager and Executor."""
        self.f_stop.set()
        for p in self.processes:
            if p.is_alive():
                p.join(timeout=3)
                p.close()
        self.manager.shutdown()
