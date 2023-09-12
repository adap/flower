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
from typing import Dict, List, Tuple, Type, cast

import ray

from flwr.client import Client, ClientState, NumPyClient
from flwr.common import Code, Config, GetPropertiesIns, GetPropertiesRes, Scalar, Status
from flwr.simulation.ray_transport.ray_actor import (
    ClientRes,
    DefaultActor,
    JobFn,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def __init__(self, cid: str) -> None:
        self.cid = int(cid)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Update state."""
        # Let's now do something with the client state
        # Let's add a new entry to the state. However, if it
        # exists we'll double its value (this tests in-memory state persistance
        # across several client spawn events -- or rounds in the case of full FL)
        if hasattr(self.state, "result_cache"):
            self.state.result_cache *= 2  # type: ignore
        else:
            self.state.result_cache = config["result"]  # type: ignore

        return {}


def get_dummy_client(cid: str) -> DummyClient:
    """Return a DummyClient."""
    return DummyClient(cid)


# A dummy workload
def job_fn(cid: str) -> JobFn:  # pragma: no cover
    """Construct a simple job with cid dependency."""

    def cid_times_pi(
        client: Client,
    ) -> ClientRes:  # pylint: disable=unused-argument
        result = int(cid) * pi

        cfg: Config = {"result": result}
        ins = GetPropertiesIns(cfg)
        client.get_properties(ins)

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

    num_proxies = 113  # a prime number
    cids = [str(cid) for cid in range(num_proxies)]

    # Prepare client states for all clients involved in the simulation
    client_states = {}
    for cid in cids:
        client_states[cid] = ClientState(cid)

    # Create actor pool
    ray.init(include_dashboard=False)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
        client_states=client_states,
    )

    # Create client proxies
    proxies = [
        RayActorClientProxy(
            client_fn=get_dummy_client,
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
    proxies, pool = prep()

    def run_once(iter_num: int) -> None:
        for prox in proxies:
            res = prox._submit_job(  # pylint: disable=protected-access
                job_fn=job_fn(prox.cid), timeout=None
            )

            res = cast(GetPropertiesRes, res)
            assert int(prox.cid) * pi == res.properties["result"]

            # Check state value
            result_cache = pool.client_states[prox.cid].result_cache  # type: ignore
            assert result_cache == int(prox.cid) * pi * iter_num

    # Submit jobs one at a time (start from uninitialised client states)
    run_once(1)
    # Submit a second time (test that client state is applied and updated fine)
    run_once(2)

    ray.shutdown()


def test_cid_consistency_all_submit_first() -> None:
    """Test that ClientProxies get the result of client job they submit.

    All jobs are submitted at the same time. Then fetched one at a time.
    """
    proxies, pool = prep()

    # submit all jobs (collect later)
    def submit_once() -> None:
        shuffle(proxies)
        for prox in proxies:
            job = job_fn(prox.cid)
            prox.actor_pool.submit_client_job(
                lambda a, c_fn, j_fn, c_state, cid: a.run.remote(
                    c_fn, j_fn, c_state, cid
                ),
                (prox.client_fn, job, prox.cid),
            )

    # fetch results one at a time
    def fetch_and_test_once(iter_num: int) -> None:
        shuffle(proxies)
        for prox in proxies:
            res = prox.actor_pool.get_client_result(prox.cid, timeout=None)
            res = cast(GetPropertiesRes, res)
            assert int(prox.cid) * pi == res.properties["result"]
            # Check state value
            result_cache = pool.client_states[prox.cid].result_cache  # type: ignore
            assert result_cache == int(prox.cid) * pi * iter_num

    # Submit jobs one at a time (start from uninitialised client states)
    submit_once()
    fetch_and_test_once(1)
    # Submit a second time (test that client state is applied and updated fine)
    submit_once()
    fetch_and_test_once(2)
    ray.shutdown()


def test_cid_consistency_without_proxies() -> None:
    """Test cid consistency of jobs submitted/retrieved to/from pool w/o ClientProxy."""
    proxies, pool = prep()
    num_clients = len(proxies)
    cids = [str(cid) for cid in range(num_clients)]

    # submit all jobs (collect later)
    def submit_once() -> None:
        shuffle(cids)
        for cid in cids:
            job = job_fn(cid)
            pool.submit_client_job(
                lambda a, c_fn, j_fn, c_state, cid_: a.run.remote(
                    c_fn, j_fn, c_state, cid_
                ),
                (get_dummy_client, job, cid),
            )

    # fetch results one at a time
    def fetch_and_test_once(iter_num: int) -> None:
        shuffle(cids)
        for cid in cids:
            res = pool.get_client_result(cid, timeout=None)
            res = cast(GetPropertiesRes, res)
            assert int(cid) * pi == res.properties["result"]
            # Check state value
            result_cache = pool.client_states[cid].result_cache  # type: ignore
            assert result_cache == int(cid) * pi * iter_num

    # Submit jobs one at a time (start from uninitialised client states)
    submit_once()
    fetch_and_test_once(1)
    # Submit a second time (test that client state is applied and updated fine)
    submit_once()
    fetch_and_test_once(2)

    ray.shutdown()
