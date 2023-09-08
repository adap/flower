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
from typing import Callable, List, Tuple, Type, cast

import ray

from flwr.client import NumPyClient
from flwr.common import Code, GetPropertiesRes, Status
from flwr.simulation.ray_transport.ray_actor import (
    ClientRes,
    DefaultActor,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy


# A dummy workload
def job_fn(cid: str) -> Callable[[], ClientRes]:  # pragma: no cover
    """Construct a simple job with cid dependency."""

    def cid_times_pi() -> ClientRes:
        result = int(cid) * pi

        # now let's convert it to a GetPropertiesRes response
        return GetPropertiesRes(
            status=Status(Code(0), message="test"), properties={"result": result}
        )

    return cid_times_pi


def prep(
    actor_type: Type[VirtualClientEngineActor] = DefaultActor,
) -> Tuple[List[RayActorClientProxy], VirtualClientEngineActorPool]:  # pragma: no cover
    """Prepare ClientProxies and pool for tests."""
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    def create_actor_fn() -> Type[VirtualClientEngineActor]:
        return actor_type.options(**client_resources).remote()  # type: ignore

    # Create actor pool
    ray.init(include_dashboard=False)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    def dummy_client(cid: str) -> NumPyClient:  # pylint: disable=unused-argument
        return NumPyClient()

    # Create 373 client proxies
    num_proxies = 373  # a prime number
    proxies = [
        RayActorClientProxy(
            client_fn=dummy_client,
            cid=str(cid),
            actor_pool=pool,
        )
        for cid in range(num_proxies)
    ]

    return proxies, pool


def test_cid_consistency_one_at_a_time() -> None:
    """Test that ClientProxies get the result of client job they submit.

    Submit one job and waits for completion. Then submits the next and so on
    """
    proxies, _ = prep()
    # submit jobs one at a time
    for prox in proxies:
        res = prox._submit_job(  # pylint: disable=protected-access
            job_fn=job_fn(prox.cid), timeout=None
        )

        res = cast(GetPropertiesRes, res)
        assert int(prox.cid) * pi == res.properties["result"]

    ray.shutdown()


def test_cid_consistency_all_submit_first() -> None:
    """Test that ClientProxies get the result of client job they submit.

    All jobs are submitted at the same time. Then fetched one at a time.
    """
    proxies, _ = prep()

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
        res = cast(GetPropertiesRes, res)
        assert int(prox.cid) * pi == res.properties["result"]

    ray.shutdown()


def test_cid_consistency_without_proxies() -> None:
    """Test cid consistency of jobs submitted/retrieved to/from pool w/o ClientProxy."""
    proxies, pool = prep()
    num_clients = len(proxies)
    cids = [str(cid) for cid in range(num_clients)]

    # submit all jobs (collect later)
    shuffle(cids)
    for cid in cids:
        job = job_fn(cid)
        pool.submit_client_job(lambda a, v, cid_: a.run.remote(v, cid_), (job, cid))

    # fetch results one at a time
    shuffle(cids)
    for cid in cids:
        res = pool.get_client_result(cid, timeout=None)
        res = cast(GetPropertiesRes, res)
        assert int(cid) * pi == res.properties["result"]

    ray.shutdown()
