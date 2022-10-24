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
from flwr.client import Client, ClientLike, to_client
from flwr.common.logger import log
from flwr.monitoring.profiler import SystemMonitor
from flwr.simulation.ray_transport.ray_client_proxy import (
    ClientFn,
    RayClientProxy,
    _create_client,
)

RayProfFunDec = TypeVar(
    "RayProfFunDec",
    bound=Callable[
        [
            Union[
                common.GetPropertiesIns,
                common.GetParametersIns,
                common.FitIns,
                common.EvaluateIns,
            ],
            Optional[float],
        ],
        Union[
            common.GetPropertiesRes,
            common.GetParametersRes,
            common.FitRes,
            common.EvaluateRes,
        ],
    ],
)


@ray.remote
class RaySystemMonitor(SystemMonitor):
    def __init__(self, *, node_id: str, interval: float = 0.1):
        super().__init__(node_id=node_id, interval=interval)


class RayClientProfilerProxy(RayClientProxy):
    def __init__(self, client_fn: ClientFn, cid: str, resources: Dict[str, float]):
        super().__init__(client_fn, cid, resources)

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        return super().get_properties(ins, timeout)

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        return super().get_parameters(ins, timeout)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
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

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        return super().evaluate(ins, timeout)


@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Execute fit remotely."""
    this_node_id = ray.get_runtime_context().get_node_id()
    this_task_id = ray.get_runtime_context().get_task_id() or "no_task_id"
    task = (this_task_id, os.getpid(), "fit")
    ray_sysmon = ray.get_actor(f"{this_node_id}", namespace="raysysmon")

    # Register task
    future_update_monitor_task_list = ray_sysmon.register_tasks.remote(tasks=[task])
    ray.get(future_update_monitor_task_list)
    client: Client = _create_client(client_fn, cid)
    result = client.fit(fit_ins)

    # Unregister task
    future_update_monitor_task_list = ray_sysmon.unregister_tasks.remote(tasks=[task])
    ray.get(future_update_monitor_task_list)
    return result


"""
def virtual_profiler(_func: RayProfFunDec) -> RayProfFunDec:
    @wraps(_func)
    def wrapper(*args, **kwargs):
        this_node_id = ray.get_runtime_context().get_node_id()
        this_task_id = ray.get_runtime_context().get_task_id() or "no_id_provided"
        task_args = (this_task_id, os.getpid(), "fit")
        ray_sysmon = ray.get_actor(f"RaySystemMonitor_{this_node_id}")
        # include task
        future_update_monitor_task_list = ray_sysmon.register_tasks.remote(
            tasks=[task_args]
        )
        ray.get(future_update_monitor_task_list)
        result = _func(*args, **kwargs)
        # remove task
        future_update_monitor_task_list = ray_sysmon.unregister_tasks.remote(
            tasks=[task_args]
        )
        ray.get(future_update_monitor_task_list)

        return result

    return cast(RayProfFunDec, wrapper)
"""
