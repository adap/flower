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


from logging import ERROR, INFO
from typing import Dict, Optional, Tuple

import flwr
from flwr.client.client import Client
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation.ray_simulation.ray_client_proxy import RayClientProxy

RAY_IMPORT_ERROR: str = """Unable to import module `ray`.

To install the necessary dependencies, install `flwr` with the `simulation` extra:

    pip install -U flwr["simulation"]
"""


def start_ray_simulation(
    pool_size: int,  # number of total partitions/clients
    data_partitions_dir: str,  # path where data partitions for each client exist
    client_resources: Dict[str, int],  # compute/memory resources for each client
    client_type: Client,
    ray_init_config: Dict = {},
    config: Optional[Dict[str, int]] = None,
    strategy: Optional[Strategy] = None,
) -> None:
    """Start a Ray-based Flower simulation server.

    Parameters
    ----------
    config: Optional[Dict[str, int]] (default: None).
        The only currently supported values is `num_rounds`, so a full
        configuration object instructing the server to perform three rounds of
        federated learning looks like the following: `{"num_rounds": 3}`.
    strategy: Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    """
    # Try to import ray
    try:
        import ray
    except:
        ray = None
    if not ray:
        log(ERROR, RAY_IMPORT_ERROR)
        return None

    # Initialize Ray
    ray.init(**ray_init_config)
    log(
        INFO,
        f"Ray initialized with resources: {ray.cluster_resources()}",
    )

    # Initialize server and server config
    initialized_server, initialized_config = flwr.server.app._init_defaults(
        None, config, strategy
    )
    log(
        INFO,
        f"Starting Flower Ray simulation running: {initialized_config}",
    )

    # Ask Ray to create and register RayClientProxy objects with the ClientManager
    for i in range(pool_size):
        client_proxy = RayClientProxy(
            cid=str(i),
            client_type=client_type,
            resources=client_resources,
            fed_dir=data_partitions_dir,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    flwr.server.app._fl(server=initialized_server, config=initialized_config)
