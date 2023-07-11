# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Ray-based Flower Actor and ActorPool implementation."""

import traceback
from typing import Any, Callable

import ray
from ray.util.actor_pool import ActorPool


class ClientException(Exception):
    """Raised when client side logic crashes with an exception."""

    def __init__(self, message):
        self.message = f"\n{'>'*7} A ClientException occurred." + message
        super().__init__(self.message)


@ray.remote
class VirtualClientEngineActor:
    """A Ray Actor class that runs client workloads."""

    def __init__(self, actor_id: int):
        self.actor_id = actor_id

    def run(self, client_fn, client_id):
        """Run a client workload."""
        # execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            client_results = client_fn()
        except Exception as ex:
            client_trace = traceback.format_exc()
            message = (
                "\n\tSomething went wrong when running your client workload."
                f"\n\tClient {client_id} crashed when the {self.__class__.__name__}"
                " was running its workload."
                f"\n\tException triggered on the client side: {client_trace}"
            )
            raise ClientException(message) from ex

        return client_id, client_results


class VirtualClientEngineActorPool(ActorPool):
    """A pool of VirtualClientEngine Actors."""

    def __init__(self, actors: list):
        super().__init__(actors)

        # stores completed job while keeping track to
        # which VirtualClient it belongs to
        self._results = {}

    def submit_client_job(self, fn: Any, value: Callable, cid: int):
        """Submit a job to the pool."""
        self._results[cid] = None
        # print(f"Submitted to pool from VirtualClient {cid}")
        return super().submit(fn, value)

    def get_client_result(self, cid: int, timeout: int = 3600):
        """Fetch the result submitted by the specified VirtualClient."""
        while self._results[cid] is None:
            # we need a try/except because if all VirtualClients are pinging the queue
            # for the first result, only one of them will be able to "fetch" it.
            # Leaving the rest trying to fetch the object following a reference that
            # doesn't exist anymore.
            try:
                res_cid, res = self.get_next_unordered(timeout=timeout)
                # Track in dictionary
                # print(f"Adding result to dict for cid: {res_cid}")
                self._results[res_cid] = res
            except KeyError:
                # result was already fetched, that's fine, continue
                continue

        if self._results[cid] is None:
            raise RuntimeError(
                f"No result is available in the VirtualClientEngine for client {cid}"
            )

        return self._results.pop(cid)
