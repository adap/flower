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

import pathlib
from logging import DEBUG, ERROR
from typing import Callable, Dict, List, Tuple, Union

import ray

from flwr.client.client_app import ClientApp
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message
from flwr.common.typing import ConfigsRecordValues
from flwr.simulation.ray_transport.ray_actor import BasicActorPool, ClientAppActor
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from .backend import Backend, BackendConfig

ClientResourcesDict = Dict[str, Union[int, float]]
ActorArgsDict = Dict[str, Union[int, float, Callable[[], None]]]
RunTimeEnvDict = Dict[str, Union[str, List[str]]]


class RayBackend(Backend):
    """A backend that submits jobs to a `BasicActorPool`."""

    def __init__(
        self,
        backend_config: BackendConfig,
        work_dir: str,
    ) -> None:
        """Prepare RayBackend by initialising Ray and creating the ActorPool."""
        log(DEBUG, "Initialising: %s", self.__class__.__name__)
        log(DEBUG, "Backend config: %s", backend_config)

        if not pathlib.Path(work_dir).exists():
            raise ValueError(f"Specified work_dir {work_dir} does not exist.")

        # Initialise ray
        self.init_args_key = "init_args"
        self.init_ray(backend_config, work_dir)

        # Validate client resources
        self.client_resources_key = "client_resources"
        client_resources = self._validate_client_resources(config=backend_config)

        # Create actor pool
        actor_kwargs = self._validate_actor_arguments(config=backend_config)

        self.pool = BasicActorPool(
            actor_type=ClientAppActor,
            client_resources=client_resources,
            actor_kwargs=actor_kwargs,
        )

    def _configure_runtime_env(self, work_dir: str) -> RunTimeEnvDict:
        """Return list of files/subdirectories to exclude relative to work_dir.

        Without this, Ray will push everything to the Ray Cluster.
        """
        runtime_env: RunTimeEnvDict = {"working_dir": work_dir}

        excludes = []
        path = pathlib.Path(work_dir)
        for p in path.rglob("*"):
            # Exclude files need to be relative to the working_dir
            if p.is_file() and not str(p).endswith(".py"):
                excludes.append(str(p.relative_to(path)))
        runtime_env["excludes"] = excludes

        return runtime_env

    def _validate_client_resources(self, config: BackendConfig) -> ClientResourcesDict:
        client_resources_config = config.get(self.client_resources_key)
        client_resources: ClientResourcesDict = {}
        valid_types = (int, float)
        if client_resources_config:
            for k, v in client_resources_config.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"client resources keys are expected to be `str` but you used "
                        f"{type(k)} for `{k}`"
                    )
                if not isinstance(v, valid_types):
                    raise ValueError(
                        f"client resources are expected to be of type {valid_types} "
                        f"but found `{type(v)}` for key `{k}`",
                    )
                client_resources[k] = v

        else:
            client_resources = {"num_cpus": 2, "num_gpus": 0.0}
            log(
                DEBUG,
                "`%s` not specified in backend config. Applying default setting: %s",
                self.client_resources_key,
                client_resources,
            )

        return client_resources

    def _validate_actor_arguments(self, config: BackendConfig) -> ActorArgsDict:
        actor_args_config = config.get("actor", False)
        actor_args: ActorArgsDict = {}
        if actor_args_config:
            use_tf = actor_args.get("tensorflow", False)
            if use_tf:
                actor_args["on_actor_init_fn"] = enable_tf_gpu_growth
        return actor_args

    def init_ray(self, backend_config: BackendConfig, work_dir: str) -> None:
        """Intialises Ray if not already initialised."""
        if not ray.is_initialized():
            # Init ray and append working dir if needed
            runtime_env = (
                self._configure_runtime_env(work_dir=work_dir) if work_dir else None
            )

            ray_init_args: Dict[
                str,
                Union[ConfigsRecordValues, RunTimeEnvDict],
            ] = {}
            
            if backend_config.get(self.init_args_key):
                for k, v in backend_config[self.init_args_key].items():
                    ray_init_args[k] = v

            if runtime_env is not None:
                ray_init_args["runtime_env"] = runtime_env

            ray.init(**ray_init_args)

    @property
    def num_workers(self) -> int:
        """Return number of actors in pool."""
        return self.pool.num_actors

    def is_worker_idle(self) -> bool:
        """Report whether the pool has idle actors."""
        return self.pool.is_actor_available()

    async def build(self) -> None:
        """Build pool of Ray actors that this backend will submit jobs to."""
        await self.pool.add_actors_to_pool(self.pool.actors_capacity)
        log(DEBUG, "Constructed ActorPool with: %i actors", self.pool.num_actors)

    async def process_message(
        self,
        app: Callable[[], ClientApp],
        message: Message,
        context: Context,
    ) -> Tuple[Message, Context]:
        """Run ClientApp that process a given message.

        Return output message and updated context.
        """
        partition_id = message.metadata.partition_id

        try:
            # Submit a task to the pool
            future = await self.pool.submit(
                lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
                (app, message, str(partition_id), context),
            )

            await future
            # Fetch result
            (
                out_mssg,
                updated_context,
            ) = await self.pool.fetch_result_and_return_actor_to_pool(future)

            return out_mssg, updated_context

        except Exception as ex:
            log(
                ERROR,
                "An exception was raised when processing a message by %s",
                self.__class__.__name__,
            )
            # add actor back into pool
            await self.pool.add_actor_back_to_pool(future)
            raise ex

    async def terminate(self) -> None:
        """Terminate all actors in actor pool."""
        await self.pool.terminate_all_actors()
        ray.shutdown()
        log(DEBUG, "Terminated %s", self.__class__.__name__)
