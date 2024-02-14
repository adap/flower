# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Fleet VirtualClientEngine API."""

import asyncio
from logging import INFO
from typing import Callable, Dict, List, Union

import ray

from flwr.client.clientapp import ClientApp, load_client_app
from flwr.client.message_handler.task_handler import configure_task_res
from flwr.client.node_state import NodeState
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.serde import message_to_taskres, recordset_from_proto
from flwr.proto.fleet_pb2 import CreateNodeRequest
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskIns
from flwr.server.superlink.fleet.message_handler.message_handler import create_node
from flwr.server.superlink.state import StateFactory
from flwr.simulation.ray_transport.ray_actor import (
    DefaultActor,
    VirtualClientEngineActorPool,
)

TaskInsQueue = asyncio.Queue[TaskIns]


def _construct_actor_pool(
    client_resources: Dict[str, Union[float, int]]
) -> VirtualClientEngineActorPool:
    """Prepare ActorPool."""

    def _create_actor_fn():  # type: ignore
        return DefaultActor.options(**client_resources).remote()  # type: ignore

    # Create actor pool
    ray.init(include_dashboard=True)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=_create_actor_fn,
        client_resources=client_resources,
    )
    return pool


def taskins_to_message(taskins: TaskIns) -> Message:
    """Convert TaskIns to Messsage."""
    recordset = recordset_from_proto(taskins.task.recordset)

    return Message(
        content=recordset,
        metadata=Metadata(
            run_id=taskins.run_id,
            message_id=taskins.task_id,
            group_id=taskins.group_id,
            node_id=0,  # TODO: resolve
            ttl="",
            message_type=taskins.task.task_type,
        ),
    )


def _register_nodes(num_nodes: int, state_factory: StateFactory) -> List[Node]:
    nodes = []
    for _ in range(num_nodes):
        node = create_node(
            request=CreateNodeRequest(), state=state_factory.state()
        ).node
        nodes.append(node)
    return nodes


async def worker(
    app: Callable[[], ClientApp],
    queue: TaskInsQueue,
    node_states: Dict[int, NodeState],
    state_factory: StateFactory,
    pool: VirtualClientEngineActorPool,
) -> None:
    """Get TaskIns from queue and execute associated job if actor is free.

    If actor is not free, the request is added back to the queue.
    """
    while True:
        task_ins = await queue.get()

        # TODO: check if another request for the same node is being running atm
        # TODO: Else potential problesm with run_state ?

        if not pool.is_actor_available():
            # TODO: revisit
            # insert actor in pool, then what?
            raise RuntimeError("Did an actor die?")

        node_id = task_ins.task.consumer.node_id

        # Register and retrive runstate
        node_states[node_id].register_context(run_id=task_ins.run_id)
        run_state = node_states[node_id].retrieve_context(run_id=task_ins.run_id)

        # Convert TaskIns to Message
        message = taskins_to_message(task_ins)

        # Submite a task to the pool
        future = pool.submit_if_actor_is_free(
            lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
            (app, message, str(node_id), run_state),
        )

        assert future is not None, "this shouldn't happen given the check above, right?"
        # print(f"wait for {future = }")
        await asyncio.wait([future])
        # print(f"got: {future = }")

        # Fetch result
        out_mssg, updated_context = pool.fetch_result_and_return_actor(future)

        # Update Context
        node_states[node_id].update_context(task_ins.run_id, context=updated_context)

        # Convert to TaskRes
        task_res = message_to_taskres(out_mssg)
        # Configuring task
        task_res = configure_task_res(
            task_res, task_ins, Node(node_id=task_ins.task.consumer.node_id)
        )
        # Store TaskRes in state
        state_factory.state().store_task_res(task_res)


async def generate_pull_requests(
    queue: TaskInsQueue,
    state_factory: StateFactory,
    nodes: List[Node],
) -> None:
    """Generate PullTaskInsRequests and adds it to the queue."""
    while True:
        for node in nodes:
            task_ins = state_factory.state().get_task_ins(node_id=node.node_id, limit=1)
            if task_ins:
                await queue.put(task_ins[0])
        await asyncio.sleep(0.5)


async def run(
    app: Callable[[], ClientApp],
    nodes: List[Node],
    state_factory: StateFactory,
    pool: VirtualClientEngineActorPool,
    node_states: Dict[int, NodeState],
) -> None:
    """Run the VCE async."""
    queue: TaskInsQueue = asyncio.Queue(64)

    worker_tasks = [
        asyncio.create_task(worker(app, queue, node_states, state_factory, pool))
        for _ in range(pool.num_actors)
    ]
    asyncio.create_task(generate_pull_requests(queue, state_factory, nodes))
    await queue.join()
    await asyncio.gather(*worker_tasks)


def run_vce(
    num_supernodes: int,
    client_resources: Dict[str, Union[float, int]],
    client_app_callable_str: str,
    state_factory: StateFactory,
) -> None:
    """Run VirtualClientEnginge."""
    # Create actor pool
    log(INFO, f"{client_resources = }")
    pool = _construct_actor_pool(client_resources)
    log(INFO, f"Constructed ActorPool with: {pool.num_actors} actors")

    # Register nodes (as many as number of possible clients)
    # Each node has its own state
    node_states: Dict[int, NodeState] = {}
    nodes = _register_nodes(num_nodes=num_supernodes, state_factory=state_factory)
    for node in nodes:
        node_states[node.node_id] = NodeState()

    log(INFO, f"Registered {len(nodes)} nodes")

    # TODO: handle different workdir
    print(f"{client_app_callable_str = }")

    def _load() -> ClientApp:
        app: ClientApp = load_client_app(client_app_callable_str)
        return app

    app = _load

    asyncio.run(run(app, nodes, state_factory, pool, node_states))
