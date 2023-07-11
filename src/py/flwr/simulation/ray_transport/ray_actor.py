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
    def __init__(self, actor_id: int):
        self.actor_id = actor_id

    def run(self, client_fn, client_id):
        # execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            client_results = client_fn()
        except Exception as ex:
            client_trace = traceback.format_exc()
            message = (
                "\n\tSomething went wrong when running your client workload."
                f"\n\tClient {client_id} crashed when the {self.__class__.__name__} was running its workload."
                f" \n\tThis is the exception triggered on the client side: {client_trace}"
            )
            raise ClientException(message)

        return client_id, client_results


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

    def get_client_result(self, cid: int, timeout: int = 3600):
        while self._results[cid] is None:
            # we need a try/except because if all VirtualClients are pinging the queue for the first
            # result, only one of them will be able to "fetch" it. Leaving the rest trying to fetch
            # the object following a reference that doesn't exist anymore.
            try:
                res_cid, res = self.get_next_unordered(timeout=timeout)
                # Track in dictionary
                # print(f"Adding result to dict for cid: {res_cid}")
                self._results[res_cid] = res
            except KeyError as ex:
                # result was already fetched, that's fine, continue
                continue

            # print(f"VirtualClient with cid: {cid} still needs to wait for its result...")
            # ready = not(self._results[cid] is None)
            # print(f"is result for {cid} read?: {ready}")

        # print(f"Returning result for client {cid}, is it none: {self._results[cid] is None}")
        if self._results[cid] is None:
            raise RuntimeError(
                f"Client {cid} failed, no result is available in the VirtualClientEngine."
            )

        return self._results.pop(cid)
