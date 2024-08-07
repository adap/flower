# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Ray backend for the Fleet API using the Simulation Engine."""

from logging import DEBUG, ERROR, WARN
from typing import Callable, Dict, Tuple, Union

import ray

from flwr.client.client_app import ClientApp
from flwr.common.constant import PARTITION_ID_KEY
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message
from flwr.simulation.ray_transport.ray_actor import BasicActorPool, ClientAppActor

from .backend import Backend
from .backendconfig import BackendConfig

ActorArgsDict = Dict[str, Union[int, float, Callable[[], None]]]


class RayBackend(Backend):
    """A backend that submits jobs to a `BasicActorPool`."""

    def __init__(
        self,
        backend_config: BackendConfig,
    ) -> None:
        """Prepare RayBackend by initialising Ray and creating the ActorPool."""
        log(DEBUG, "Initialising: %s", self.__class__.__name__)
        log(DEBUG, "Backend config: %s", backend_config)

        # Proactively silence INFO logs
        if "logging_level" not in backend_config.config:
            backend_config.config["logging_level"] = WARN

        # Initialise ray
        self.init_ray(backend_config)

        # Express ClientApp resources in a way ray understands
        client_resources = {
            "num_cpus": backend_config.clientapp_resources.num_cpus,
            "num_gpus": backend_config.clientapp_resources.num_gpus,
        }

        # Create actor pool
        self.pool = BasicActorPool(
            actor_type=ClientAppActor,
            client_resources=client_resources,
            actor_kwargs={},
        )

    def init_ray(self, backend_config: BackendConfig) -> None:
        """Intialises Ray if not already initialised."""
        if not ray.is_initialized():
            ray_init_args = backend_config.config

            log(DEBUG, "Initializing Ray with passed config: %s", ray_init_args)
            ray.init(**ray_init_args)

    @property
    def num_workers(self) -> int:
        """Return number of actors in pool."""
        return self.pool.num_actors

    def is_worker_idle(self) -> bool:
        """Report whether the pool has idle actors."""
        return self.pool.is_actor_available()

    def build(self) -> None:
        """Build pool of Ray actors that this backend will submit jobs to."""
        self.pool.add_actors_to_pool(self.pool.actors_capacity)
        log(DEBUG, "Constructed ActorPool with: %i actors", self.pool.num_actors)

    def process_message(
        self,
        app: Callable[[], ClientApp],
        message: Message,
        context: Context,
    ) -> Tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        partition_id = context.node_config[PARTITION_ID_KEY]

        try:
            # Submit a task to the pool
            future = self.pool.submit(
                lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
                (app, message, str(partition_id), context),
            )

            # Fetch result
            (
                out_mssg,
                updated_context,
            ) = self.pool.fetch_result_and_return_actor_to_pool(future)

            return out_mssg, updated_context

        except Exception as ex:
            log(
                ERROR,
                "An exception was raised when processing a message by %s",
                self.__class__.__name__,
            )
            # add actor back into pool
            self.pool.add_actor_back_to_pool(future)
            raise ex

    def terminate(self) -> None:
        """Terminate all actors in actor pool."""
        self.pool.terminate_all_actors()
        ray.shutdown()
        log(DEBUG, "Terminated %s", self.__class__.__name__)
