# Copyright 2023 Flower Labs. All Rights Reserved.
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


import threading
import traceback
from abc import ABC
from logging import ERROR, WARNING
from typing import Any, Callable, Dict, List, Set, Union

import ray
from ray.util.actor_pool import ActorPool

from flwr.common.logger import log


class ClientException(Exception):
    """Raised when client side logic crashes with an exception."""

    def __init__(self, message: str):
        self.message = f"\n{'>'*7} A ClientException occurred." + message
        super().__init__(self.message)


class VirtualClientEngineActor(ABC):
    """Abstract base class for VirtualClientEngine Actors"""

    def terminate(self):
        """Manually terminate Actor object."""
        log(WARNING, f"Manually terminating {self.__class__.__name__}")
        ray.actor.exit_actor()
    
    def run(self, job_fn: Callable, cid: str):
        """Run a client workload."""
        # execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            job_results = job_fn()
        except Exception as ex:
            client_trace = traceback.format_exc()
            message = (
                "\n\tSomething went wrong when running your client workload."
                f"\n\tClient {cid} crashed when the {self.__class__.__name__}"
                " was running its workload."
                f"\n\tException triggered on the client side: {client_trace}"
            )
            raise ClientException(message) from ex

        return cid, job_results


@ray.remote
class DefaultActor(VirtualClientEngineActor):
    """A Ray Actor class that runs client workloads."""


@ray.remote
class DefaultActor_TF(VirtualClientEngineActor):
    """A Ray Actor class that runs TF client workloads.
    
    It enables GPU memory growth to prevent premature OOM."""

    def __init__(self):
        super().__init__()
        # By default, TF maps all GPU memory to the process.
        # We don't this behaviour in simulation, since it prevents us
        # from having multiple Actors (and therefore Flower clients) sharing
        # the same GPU.
        # Luckily we can disable this behaviour by enabling memory growth
        # on the GPU. In this way, VRAM allocated to the processes grows based
        # on the needs for the workload. (this is for instance the default 
        # behaviour in Pytorch)
        try:
            import tensorflow as tf
            # this bit of code follows the guidelines for GPU usage
            # in https://www.tensorflow.org/guide/gpu
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        except Exception as e:
            log(ERROR, "Do you have Tensorflow installed?")
            raise e



def pool_size_from_resources(client_resources: Dict):
    """Calculate number of Actors that fit in pool given the resources in the.

    cluster and those required per client.
    """
    cluster_resources = ray.cluster_resources()
    num_cpus = cluster_resources["CPU"]
    num_gpus = cluster_resources.get("GPU", 0)  # there might not be GPU
    num_actors = int(num_cpus / client_resources["num_cpus"])
    # if a GPU is present and client resources do require one
    if "num_gpus" in client_resources.keys() and client_resources["num_gpus"] > 0.0:
        if num_gpus:
            # if there are gpus in the cluster
            num_actors = min(num_actors, int(num_gpus / client_resources["num_gpus"]))
        else:
            num_actors = 0

    if num_actors == 0:
        log(
            WARNING,
            f"Your ActorPool is empty. Your system ({num_cpus = }, {num_gpus = }) "
            "does not meet the criteria to host at least one client with resources:"
            f" {client_resources}. Consider lowering your `client_resources`",
        )
        raise ValueError(f"ActorPool is empty. Stopping Simulation." \
                         "Check 'client_resources'")

    return num_actors


class VirtualClientEngineActorPool(ActorPool):
    """A pool of VirtualClientEngine Actors."""

    def __init__(self,
                 client_resources: Dict[str, Union[int, float]],
                 actor_type: VirtualClientEngineActor,
                 actor_kwargs: Dict[str, Any],
                 max_restarts: int,
                 ):
        self.client_resources = client_resources
        self.actor_type = actor_type
        self.actor_kwargs = actor_kwargs
        self.actor_max_restarts = max_restarts
        num_actors = pool_size_from_resources(client_resources)
        actors = [
            actor_type.options(
                **client_resources, max_restarts=max_restarts
            ).remote(**actor_kwargs)
            for _ in range(num_actors)
        ]

        super().__init__(actors)

        # A dict that maps cid to another dict containing: a reference to the remote job
        # and its status (i.e. whether it is ready or not)
        self._cid_to_future: Dict[str, Dict[str, Any]] = {}
        self.actor_to_remove: Set[str] = set()  # a set
        self.num_actors = len(actors)

        self.lock = threading.RLock()

        # TODO: asyncio check every N seconds if cluster has grown
        # --> add more actors to the pool if so

    def __reduce__(self):
        """Make this class serialisable (needed due to lock)."""
        return VirtualClientEngineActorPool, (
            self.client_resources,
            self.actor_type,
            self.actor_kwargs,
            self.actor_max_restarts,
        )

    def submit(self, fn: Any, job_fn: Callable, cid: str) -> None:
        """Take idle actor and assign it a client workload."""
        actor = self._idle_actors.pop()
        if self._check_and_remove_actor_from_pool(actor):
            future = fn(actor, job_fn)
            future_key = tuple(future) if isinstance(future, List) else future
            self._future_to_actor[future_key] = (self._next_task_index, actor, cid)
            self._next_task_index += 1

            # update with future
            self._cid_to_future[cid]["future"] = future_key

    def submit_client_job(self, fn: Any, job_fn: Callable, cid: str) -> None:
        """Submit a job while tracking client ids."""
        # We need to put this behind a lock since .submit() involves
        # removing and adding elements from a dictionary. Which creates
        # issues in multi-threaded settings
        with self.lock:
            # creating cid to future mapping
            self._reset_cid_to_future_dict(cid)
            if self._idle_actors:
                # submit job since there is an Actor that's available
                self.submit(fn, job_fn, cid)
            else:
                # no actors are available, append to list of jobs to run later
                self._pending_submits.append((fn, job_fn, cid))

    def _flag_future_as_ready(self, cid) -> None:
        """Flag future for VirtualClient with cid=cid as ready."""
        self._cid_to_future[cid]["ready"] = True

    def _reset_cid_to_future_dict(self, cid: str) -> None:
        """Reset cid:future mapping info."""
        if cid not in self._cid_to_future.keys():
            self._cid_to_future[cid] = {}

        self._cid_to_future[cid]["future"] = None
        self._cid_to_future[cid]["ready"] = False

    def _is_future_ready(self, cid: str) -> bool:
        """Return status of future associated to the given client id (cid)."""
        if cid not in self._cid_to_future.keys():
            return False
        else:
            return self._cid_to_future[cid]["ready"]

    def _fetch_future_result(self, cid: str) -> Any:
        """Fetch result for VirtualClient from Object Store."""
        try:
            res_cid, res = ray.get(self._cid_to_future[cid]["future"])
        except ray.exceptions.RayActorError as ex:
            log(ERROR, ex)
            if hasattr(ex, "actor_id"):
                # RayActorError only contains the actor_id attribute
                # if the actor won't be restarted again.
                self._flag_actor_for_removal(ex.actor_id)
            raise ex

        # sanity check: was the result fetched generated by a client with cid=cid?
        assert res_cid != res, log(
            ERROR, f"The VirtualClient {cid} got result from client {res_cid}"
        )

        # reset mapping
        self._reset_cid_to_future_dict(cid)

        return res

    def _flag_actor_for_removal(self, actor_id_hex: str) -> None:
        """Flag actor that should be removed from pool."""
        with self.lock:
            self.actor_to_remove.add(actor_id_hex)
            log(WARNING, f"Actor({actor_id_hex}) will be remove from pool.")

    def _check_and_remove_actor_from_pool(
        self, actor: VirtualClientEngineActor
    ) -> bool:
        """Check if actor in set of those that should be removed.

        Remove the actor if so.
        """
        with self.lock:
            actor_id = actor._actor_id.hex()
            # print(f"{self.actor_to_remove = }")
            if actor_id in self.actor_to_remove:
                # the actor should be removed
                self.actor_to_remove.remove(actor_id)
                self.num_actors -= 1
                log(WARNING, f"REMOVED actor {actor_id} from pool")
                log(WARNING, f"Pool size: {self.num_actors}")
                return False
            else:
                return True

    def _check_actor_fits_in_pool(self) -> bool:
        """Determine if available resources haven't changed.

        If true, allow the actor to be added back to the pool. Else don't allow it
        (effectively reducing the size of the pool).
        """
        num_actors_updated = pool_size_from_resources(self.client_resources)

        if num_actors_updated < self.num_actors:
            log(
                WARNING,
                "Cluster resources have changed. Number of actors in the pool should be"
                f" reduced from {self.num_actors} down to {num_actors_updated}. This"
                " might take several intermediate steps",
            )
            # we are preventing one actor to be added back in the queue, so we just
            # decrease the number of actors by one. Eventually `self.num_actors`
            # should be equal what pool_size_from_resources(self.resources) returns
            self.num_actors -= 1
            return False
        else:
            return True

    def process_unordered_future(self, timeout=None) -> None:
        """Similar to parent's get_next_unordered() but without final ray.get()."""
        if not self.has_next():
            raise StopIteration("No more results to get")
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)

        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")

        with self.lock:
            # get actor that completed a job
            _, a, cid = self._future_to_actor.pop(future, (None, None, -1))
            if a is not None:
                # still space in queue ? (no if a node died)
                if self._check_actor_fits_in_pool():
                    if self._check_and_remove_actor_from_pool(a):
                        self._return_actor(a)
                    # flag future as ready
                    self._flag_future_as_ready(cid)
                else:
                    # the actor doesn't fit in the pool anymore.
                    # Manually terminate the actor
                    a.terminate.remote()

    def get_client_result(self, cid: str, timeout: int = 3600) -> Any:
        """Get result from VirtualClient with specific cid."""
        # loop until all jobs submitted to the pool are completed. Break early
        # if the result for the ClientProxy calling this method is ready
        while self.has_next() and not (self._is_future_ready(cid)):
            try:
                self.process_unordered_future(timeout=timeout)
            except StopIteration:
                # there are no pending jobs in the pool
                break

        # Fetch result belonging to the VirtualClient calling this method
        return self._fetch_future_result(cid)
