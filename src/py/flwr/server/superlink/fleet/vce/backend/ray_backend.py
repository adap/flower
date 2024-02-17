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
"""Ray backend for the Fleet API using the VCE."""

import asyncio
from logging import INFO
from typing import Callable, Dict, Tuple, Union

from flwr.client.clientapp import ClientApp
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message
from flwr.simulation.ray_transport.ray_actor import (
    BasicActorPool,
    ClientAppActor,
    init_ray,
)

from .backend import Backend


class RayBackend(Backend):
    """A backend that submits jobs to a `BasicActorPool`."""

    def __init__(
        self,
        client_resources: Dict[str, Union[float, int]],
        wdir: str,
    ) -> None:
        """Prepare ActorPool."""
        log(INFO, f"{client_resources = }")
        # Init ray and append working dir if needed
        # Ref: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
        runtime_env = {"working_dir": wdir} if wdir else None
        init_ray(
            include_dashboard=True, runtime_env=runtime_env
        )  # TODO: recursiviely search dir, we don't want that. use `excludes` arg
        # Create actor pool
        self._pool = BasicActorPool(
            actor_type=ClientAppActor,
            client_resources=client_resources,
        )

    @property
    def num_workers(self) -> int:
        """Return number of actors in pool."""
        return self._pool.num_actors

    async def build(self) -> None:
        """Build pool of actors."""
        await self._pool.add_actors_to_pool(self._pool.actors_capacity)
        log(INFO, f"Constructed ActorPool with: {self._pool.num_actors} actors")

    async def process_message(
        self,
        app: Callable[[], ClientApp],
        message: Message,
        context: Context,
        node_id: int,
    ) -> Tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        assert self._pool.is_actor_available(), "This should never happen."

        # Submite a task to the pool
        future = await self._pool.submit_if_actor_is_free(
            lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
            (app, message, str(node_id), context),
        )

        assert future is not None, "this shouldn't happen given the check above, right?"
        # print(f"wait for {future = }")
        await asyncio.wait([future])
        # print(f"got: {future = }")

        # Fetch result
        out_mssg, updated_context = await self._pool.fetch_result_and_return_actor(
            future
        )

        return out_mssg, updated_context
