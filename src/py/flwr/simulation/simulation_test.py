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

from math import pi
from random import shuffle
from typing import Type

import ray

from flwr.simulation.ray_transport.ray_actor import (
    DefaultActor,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy


# A dummy workload
def job_fn(cid: str):  # pragma: no cover
    """Construct a simple job with cid dependency."""

    def cid_times_pi() -> float:
        return int(cid) * pi

    return cid_times_pi


def prep():  # pragma: no cover
    """Prepare ClientProxies and pool for tests."""
    client_resources = {"num_cpus": 1}

    def create_actor_fn() -> Type[VirtualClientEngineActor]:
        return DefaultActor.options(**client_resources).remote()

    # Create actor pool
    ray.init(include_dashboard=False)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    # # Create 1009 client proxies
    N = 1009  # a prime number
    proxies = [
        RayActorClientProxy(
            client_fn=None,
            cid=str(cid),
            actor_pool=pool,
        )
        for cid in range(N)
    ]

    return proxies


def test_cid_consistency_one_at_a_time() -> None:
    """Test that ClientProxies get the result of client job they submit."""
    proxies = prep()
    # submit jobs one at a time
    for prox in proxies:
        res = prox._submit_job(job_fn=job_fn(prox.cid), timeout=None)
        assert int(prox.cid) * pi == res

    ray.shutdown()


def test_cid_consistency_all_submit_first() -> None:
    """Test that ClientProxies get the result of client job they submit."""
    proxies = prep()

    # submit all jobs (collect later)
    shuffle(proxies)
    for prox in proxies:
        job = job_fn(prox.cid)
        prox.actor_pool.submit_client_job(
            lambda a, v, cid: a.run.remote(v, cid), (job, prox.cid)
        )

    # fetch results one at a time
    shuffle(proxies)
    for prox in proxies:
        res = prox.actor_pool.get_client_result(prox.cid, timeout=None)
        assert int(prox.cid) * pi == res

    ray.shutdown()
