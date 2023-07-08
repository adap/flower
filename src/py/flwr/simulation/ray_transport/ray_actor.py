from typing import Any, Callable

import ray
from ray.util.actor_pool import ActorPool

@ray.remote
class VirtualClientEngineActor:

    def __init__(self, actor_id: int):
        
        self.actor_id = actor_id
    
    def run(self, client_fn, client_id):
        # execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        return client_id, client_fn()


class VirtualClientEngineActorPool(ActorPool):

    def __init__(self, actors: list):
        super().__init__(actors)

        # stores completed job while keeping track to
        # which VirtualClient it belongs to
        self._results = {}

    def submit_client_job(self, fn: Any, value: Callable, cid: int):
        self._results[cid] = None
        # print(f"Submitted to pool from VirtualClient {cid}")
        return super().submit(fn, value)

    def get_client_result(self, cid: int, timeout: int=3600):

        while self._results[cid] is None:
            # we need a try/except because if all VirtualClients are pinging the queue for the first
            # result, only one of them will be able to "fetch" it. Leaving the rest trying to fetch
            # the object following a reference that doesn't exist anymore.
            try:
                res_cid, res = self.get_next_unordered()
                # Track in dictionary
                # print(f"Adding result to dict for cid: {res_cid}")
                self._results[res_cid] = res
            except:
                continue

            # print(f"VirtualClient with cid: {cid} still needs to wait for its result...")
            # ready = not(self._results[cid] is None)
            # print(f"is result for {cid} read?: {ready}")

        # print(f"Returning result for client {cid}")
        return self._results.pop(cid)
    