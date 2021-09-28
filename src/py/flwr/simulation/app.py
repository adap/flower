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
from typing import Any, Callable, Dict, List, Optional, Union

import ray

from flwr.client.client import Client
from flwr.common.logger import log
from flwr.server.app import _fl, _init_defaults
from flwr.server.strategy import Strategy
from flwr.simulation.ray_transport.ray_client_proxy import RayClientProxy


def start_simulation(  # pylint: disable=too-many-arguments
    client_fn: Callable[[str], Client],
    clients_ids: Union[int, List[str]],
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
    clients_ids : Union[int, List[str]]
        List of possible `client_id`s (if List[str]) for simulated clients or
        the total number of clients in this simulation (if int).
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
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:

            {
                "ignore_reinit_error": True,
                "include_dashboard": False,
            }

        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    """

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
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

    # Check if a number of client or a list of clients_ids was passed
    num_clients: int
    list_clients: List[str]
    if isinstance(clients_ids, int):
        num_clients = clients_ids
        list_clients = [str(x) for x in range(num_clients)]
    else:
        num_clients = len(clients_ids)
        list_clients = clients_ids

    # Register one RayClientProxy object for each client with the ClientManager
    resources = client_resources if client_resources is not None else {}
    for cid in list_clients:
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=cid,
            resources=resources,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    _fl(
        server=initialized_server,
        config=initialized_config,
        force_final_distributed_eval=False,
    )
