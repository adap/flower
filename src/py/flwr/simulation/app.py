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


import sys
from logging import ERROR, INFO
from typing import Any, Callable, Dict, List, Optional

import ray

from flwr.client import ClientLike
from flwr.common import EventType, event
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.app import ServerConfig, init_defaults, run_fl
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.simulation.ray_transport.ray_actor import VirtualClientEngineActorPool, VirtualClientEngineActor, DefaultActor
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy

INVALID_ARGUMENTS_START_SIMULATION = """
INVALID ARGUMENTS ERROR

Invalid Arguments in method:

`start_simulation(
    *,
    client_fn: Callable[[str], ClientLike],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: ServerConfig = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> None:`

REASON:
    Method requires:
        - Either `num_clients`[int] or `clients_ids`[List[str]]
        to be set exclusively.
        OR
        - `len(clients_ids)` == `num_clients`

"""


def start_simulation(  # pylint: disable=too-many-arguments
    *,
    client_fn: Callable[[str], ClientLike],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
    actor_type: Optional[VirtualClientEngineActor] = DefaultActor,
    actor_kwargs: Optional[Dict[str, Any]] = {},
) -> History:
    """Start a Ray-based Flower simulation server.

    Parameters
    ----------
    client_fn : Callable[[str], ClientLike]
        A function creating client instances. The function must take a single
        `str` argument called `cid`. It should return a single client instance
        of type ClientLike. Note that the created client instances are ephemeral
        and will often be destroyed after a single method invocation. Since client
        instances are not long-lived, they should not attempt to carry state over
        method invocations. Any state required by the instance (model, dataset,
        hyperparameters, ...) should be (re-)created in either the call to `client_fn`
        or the call to any of the client methods (e.g., load evaluation data in the
        `evaluate` method itself).
    num_clients : Optional[int]
        The total number of clients in this simulation. This must be set if
        `clients_ids` is not set and vice-versa.
    clients_ids : Optional[List[str]]
        List `client_id`s for each client. This is only required if
        `num_clients` is not set. Setting both `num_clients` and `clients_ids`
        with `len(clients_ids)` not equal to `num_clients` generates an error.
    client_resources : Optional[Dict[str, float]] (default: `{"num_cpus": 1,
        "num_gpus": 0.0}` CPU and GPU resources for a single client. Supported keys
        are `num_cpus` and `num_gpus`. To understand the GPU utilization caused by
        `num_gpus`, as well as using custom resources, please consult the Ray
        documentation.
    server : Optional[flwr.server.Server] (default: None).
        An implementation of the abstract base class `flwr.server.Server`. If no
        instance is provided, then `start_server` will create one.
    config: ServerConfig (default: None).
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class `flwr.server.ClientManager`.
        If no implementation is provided, then `start_simulation` will use
        `flwr.server.client_manager.SimpleClientManager`.
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:

        { "ignore_reinit_error": True, "include_dashboard": False }

        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    keep_initialised: Optional[bool] (default: False)
        Set to True to prevent `ray.shutdown()` in case `ray.is_initialized()=True`.
    
    actor_type: Optional[VirtualClientEngineActor] (default: DefaultActor)

        Optionally specify the type of actor to use. The actor object, which
        persist throughout the simulation will be the process in charged of
        running the clients' jobs (i.e. their fit() method). If you are using
        Tensorflow, you should use type `DefaultActor_TF` which will set TF's
        GPU memory growth to True at initialisation (preventing premature OOM).

    actor_kwargs: Optional[Dict[str, Any]] (default: {})
        If you want to create your own Actor classes, you might need to pass
        some input argument. You can use this dictionary for such purpose.

    Returns
    -------
    hist : flwr.server.history.History
        Object containing metrics from training.
    """
    # pylint: disable-msg=too-many-locals
    event(
        EventType.START_SIMULATION_ENTER,
        {"num_clients": len(clients_ids) if clients_ids is not None else num_clients},
    )

    # Initialize server and server config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )

    log(
        INFO,
        "Starting Flower simulation, config: %s",
        initialized_config,
    )

    # clients_ids takes precedence
    cids: List[str]
    if clients_ids is not None:
        if (num_clients is not None) and (len(clients_ids) != num_clients):
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION)
            sys.exit()
        else:
            cids = clients_ids
    else:
        if num_clients is None:
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION)
            sys.exit()
        else:
            cids = [str(x) for x in range(num_clients)]

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized (unless asked not to)
    if ray.is_initialized() and not keep_initialised:
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    cluster_resources = ray.cluster_resources()
    log(
        INFO,
        "Flower VCE: Ray initialized with resources: %s",
        cluster_resources,
    )

    # Log the resources that a single client will be able to use
    if client_resources is None:
        log(
            INFO,
            "No `client_resources` specified. Using minimal resources for clients.",
        )
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    log(
        INFO,
        "Flower VCE: Resources for each Virtual Client: %s",
        client_resources,
    )

    # Instantiate ActorPool
    # TODO: maybe we want `max_restarts` to be user-defined ?
    # `max_restarts` determines how many times an actor that crashes should be restarted
    # after these many restarts, it will be removed from the pool
    max_restarts = 1 

    pool = VirtualClientEngineActorPool(client_resources,
                                        actor_type,
                                        actor_kwargs,
                                        max_restarts)

    log(
        INFO,
        "Flower VCE: Creating %s with %s actors",
        pool.__class__.__name__,
        pool.num_actors,
    )

    # Register one RayClientProxy object for each client with the ClientManager
    for cid in cids:
        client_proxy = RayActorClientProxy(
            client_fn=client_fn,
            cid=cid,
            actor_pool=pool,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    try:
        hist = run_fl(
            server=initialized_server,
            config=initialized_config,
        )
    except Exception as ex:
        log(ERROR, ex)
        log(
            ERROR,
            "Your simulation crashed :(. This could be because of several reasons."
            "The most common are: "
            "\n\t > Your system couldn't fit a single VirtualClient: try lowering "
            "`client_resources`."
            "\n\t > All the actors in your pool crashed. This could be because: "
            "\n\t\t - You clients hit an out-of-memory (OOM) error and actors couldn't "
            "recover from it. Try launching your simulation with more generous "
            f"`client_resources` setting (i.e. it seems {client_resources} is "
            "not enough for your workload). Use fewer concurrent actors. "
            "\n\t\t - You were running a multi-node simulation and all worker nodes "
            "disconnected. The head node might still be alive but cannot accommodate "
            f"any actor with resources: {client_resources}."
            "\n\t\t - Your Actors crashed because of an unknown reason and all their "
            f"restarts attempts ({max_restarts=}) have been exhausted.",
        )
        hist = None

    event(EventType.START_SIMULATION_LEAVE)

    return hist
