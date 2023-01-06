# Copyright 2022 Adap GmbH. All Rights Reserved.
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
import os
from logging import DEBUG
from functools import wraps
from typing import Callable, Dict, List, Optional, TypeVar, Union, cast

import ray
from ray.experimental.state.api import list_nodes, list_tasks

from flwr import common
from flwr.client import Client
from flwr.common.logger import log
from flwr.monitoring.profiler import SystemMonitor
from flwr.simulation.ray_transport.ray_client_proxy import (
    ClientFn,
    RayClientProxy,
    _create_client,
)


@ray.remote
class RaySystemMonitor(SystemMonitor):
    def __init__(self, *, node_id: str, interval_s: float = 0.1):
        super().__init__(node_id=node_id, interval_s=interval_s)


class RayClientProfilerProxy(RayClientProxy):
    def __init__(
        self,
        client_fn: ClientFn,
        cid: str,
        resources: Dict[str, float],
    ):
        super().__init__(client_fn, cid, resources)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources
        ).remote(self.client_fn, self.cid, ins)
        try:
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        return cast(
            common.FitRes,
            res,
        )


@ray.remote(max_calls=1)
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Execute fit remotely."""

    # Get node_id and task_id
    this_node_id = ray.get_runtime_context().get_node_id()
    this_task_id = ray.get_runtime_context().get_task_id() or "no_task_id"
    task = (this_task_id, os.getpid(), f"{cid}_fit")
    sysmon = ray.get_actor(f"{this_node_id}", namespace="flwr_experiment")

    # Register task
    future_update_monitor_task_list = sysmon.register_tasks.remote(tasks=[task])
    print(f"Registered client {cid}: {ray.get(future_update_monitor_task_list)}")
    from time import sleep

    client: Client = _create_client(client_fn, cid)
    fit_res = client.fit(fit_ins)

    # Inject task_id and node_id
    fit_res.metrics["_flwr.monitoring.task_id"] = this_task_id
    fit_res.metrics["_flwr.monitoring.node_id"] = this_node_id

    # Do not unregister task, this is done by the aggregate_fit
    future_update_monitor_task_list = sysmon.unregister_tasks.remote(
        task_ids=[this_task_id]
    )
    print(f"Unregistered client {cid}: {ray.get(future_update_monitor_task_list)}")

    return fit_res
