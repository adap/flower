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
"""Flower simulation app."""


from logging import INFO
from typing import Any, Callable, Dict, Optional

import ray

from flwr.client.client import Client
from flwr.common.logger import log
from flwr.server.app import _fl, _init_defaults
from flwr.server.strategy import Strategy
from flwr.simulation.ray_transport.ray_client_proxy import RayClientProxy


def start_simulation(  # pylint: disable=too-many-arguments
    client_fn: Callable[[str], Client],
    num_clients: int,
    client_resources: Optional[Dict[str, int]] = None,
    num_rounds: int = 1,
    strategy: Optional[Strategy] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Start a Ray-based Flower simulation server.

    Parameters
    ----------
    client_fn : Callable[[str], Client]
        A function creating client instances. The function must take a single
        str argument called `cid`. It should return a single client instance.
        Note that the created client instances are ephemeral and will often be
        destroyed after a single method invocation. Since client instances are
        not long-lived, they should not attempt to carry state over method
        invocations. Any state required by the instance (model, dataset,
        hyperparameters, ...) should be (re-)created in either the call to
        `client_fn` or the call to any of the client methods (e.g., load
        evaluation data in the `evaluate` method itself).
    num_clients : int
        The total number of clients in this simulation.
    client_resources : Optional[Dict[str, int]] (default: None)
        CPU and GPU resources for a single client. Supported keys are
        `num_cpus` and `num_gpus`. Example: `{"num_cpus": 4, "num_gpus": 1}`.
        To understand the GPU utilization caused by `num_gpus`, consult the Ray
        documentation on GPU support.
    num_rounds : int (default: 1)
        The number of rounds to train.
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    ray_init_args : Optional[Dict[str, Any]] (default: {'ignore_reinit_error': True})
        Optional dictionary containing `ray.init` arguments.
    """

    # Initialize Ray
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
        }
    ray.init(**ray_init_args)
    log(
        INFO,
        "Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    # Initialize server and server config
    config = {"num_rounds": num_rounds}
    initialized_server, initialized_config = _init_defaults(None, config, strategy)
    log(
        INFO,
        "Starting Flower simulation running: %s",
        initialized_config,
    )

    # Register one RayClientProxy object for each client with the ClientManager
    resources = client_resources if client_resources is not None else {}
    for i in range(num_clients):
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=str(i),
            resources=resources,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    _fl(
        server=initialized_server,
        config=initialized_config,
        force_final_distributed_eval=False,
    )
