# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Flower simulation tests."""

from time import sleep
from math import pi
from typing import Type
from random import shuffle

import ray

from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy
from flwr.simulation.ray_transport.ray_actor import (
    DefaultActor,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
)


def test_cid_consistency() -> None:
    """Test that ClientProxies get the result of client job they submit."""

    # A dummy workload
    def job_fn(cid: str):
        def cid_times_pi() -> float:
            return int(cid) * pi
        return cid_times_pi

    client_resources = {'num_cpus': 1}

    def create_actor_fn() -> Type[VirtualClientEngineActor]:
        return DefaultActor.options(**client_resources).remote()

    # Create actor pool
    ray.init()
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    # # Create 100 client proxies
    N = 100
    proxies = [RayActorClientProxy(client_fn=None,
                                   cid=str(cid),
                                   actor_pool=pool,) for cid in range(N)]

    # submit jobs one at a time
    for prox in proxies:
        res = prox._submit_job(job_fn=job_fn(prox.cid), timeout=None)
        assert int(prox.cid) * pi == res

    # submit all jobs (collect later)
    shuffle(proxies)
    for prox in proxies:
        # print("--------------------------------", prox.cid)
        job = job_fn(prox.cid)
        prox.actor_pool.submit_client_job(
                # lambda a, v: a.run.remote(v, prox.cid), (job, prox.cid)
                lambda a, v, cid: a.run.remote(v, cid), (job, prox.cid)

            )
        
        # print(f"{prox.cid = } ----> result should be: {job() = }")

    sleep(5)
    # print(f"----------------------- FETCHING RESULTS --------------------------------")
    # fetch results one at a time
    shuffle(proxies)
    for prox in proxies:

        # print("--------------------------------", prox.cid)
        res = prox.actor_pool.get_client_result(prox.cid, timeout=None)

        # print(f"proxy {prox.cid} got result {res}")

        assert int(prox.cid) * pi == res
        # print("OK")
