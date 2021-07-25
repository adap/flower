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
"""Flower Simulation app."""


from logging import INFO
from typing import Callable, Dict, Optional

try:
    import ray
except ImportError:
    ray = None

from flwr.client.client import Client
from flwr.common.logger import log
from flwr.server.app import _fl, _init_defaults
from flwr.server.strategy import Strategy
from flwr.simulation.ray_simulation.ray_client_proxy import RayClientProxy

RAY_IMPORT_ERROR: str = """Unable to import module `ray`.

To install the necessary dependencies, install `flwr` with the `simulation` extra:

    pip install -U flwr["simulation"]
"""


def start_simulation(  # pylint: disable=too-many-arguments
    client_fn: Callable[[str], Client],
    nb_clients: int,  # number of total partitions/clients
    client_resources: Dict[str, int],  # compute/memory resources for each client
    ray_init_config: Optional[Dict[str, int]] = None,
    nb_rounds: int = 1,
    strategy: Optional[Strategy] = None,
) -> None:
    """Start a Ray-based Flower simulation server.

    Parameters
    ----------
    strategy: Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    """

    # Ray cannot be assumed to be installed
    if ray is None:
        raise ImportError(RAY_IMPORT_ERROR)

    # Initialize Ray
    if not ray_init_config:
        ray_init_config = {}
    ray.init(**ray_init_config)
    log(
        INFO,
        "Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    # Initialize server and server config
    config = {"num_rounds": nb_rounds}
    initialized_server, initialized_config = _init_defaults(None, config, strategy)
    log(
        INFO,
        "Starting Flower Ray simulation running: %s",
        initialized_config,
    )

    # Ask Ray to create and register RayClientProxy objects with the ClientManager
    for i in range(nb_clients):
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=str(i),
            resources=client_resources,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    _fl(server=initialized_server, config=initialized_config)
