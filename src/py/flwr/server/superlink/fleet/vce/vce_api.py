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
"""Fleet VirtualClientEngine API."""

import asyncio
import traceback
from logging import ERROR, INFO
from typing import Callable, Dict, Type, Union

from flwr.client.clientapp import ClientApp, load_client_app
from flwr.client.node_state import NodeState
from flwr.common.logger import log
from flwr.common.serde import message_from_taskins, message_to_taskres
from flwr.proto.task_pb2 import TaskIns
from flwr.server.superlink.state import StateFactory
from flwr.common import Metadata, Message

from .backend import Backend, RayBackend

TaskInsQueue = asyncio.Queue[TaskIns]
NodeToPartitionMapping = Dict[int, int]


supported_backends = {"ray": RayBackend}


def _register_nodes(
    num_nodes: int, state_factory: StateFactory
) -> NodeToPartitionMapping:
    nodes_mapping: NodeToPartitionMapping = {}
    for i in range(num_nodes):
        node_id = state_factory.state().create_node()
        nodes_mapping[node_id] = i
    return nodes_mapping


async def worker(
    app: Callable[[], ClientApp],
    queue: TaskInsQueue,
    node_states: Dict[int, NodeState],
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
    backend: Backend,
) -> None:
    """Get TaskIns from queue and pass it to an actor in the pool to execute it."""
    while True:
        try:
            task_ins = await queue.get()

            # TODO: check if another request for the same node is being running atm
            # TODO: Else potential problesm with run_state ?

            node_id = task_ins.task.consumer.node_id

            # Register and retrive runstate
            node_states[node_id].register_context(run_id=task_ins.run_id)
            context = node_states[node_id].retrieve_context(run_id=task_ins.run_id)

            # Convert TaskIns to Message
            message = message_from_taskins(task_ins)
            # Replace node-id with data partition id
            message.metadata.dst_node_id = nodes_mapping[node_id]

            # Let backend process message
            out_mssg, updated_context = await backend.process_message(
                app, message, context, node_id
            )

            # Update Context
            node_states[node_id].update_context(
                task_ins.run_id, context=updated_context
            )

            # TODO: can we avoid going to proto ?
            # TODO: maybe with a new StateFactory + In-Memory Driver-SuperLink conn.
            # Undo change node_id for partition choice
            out_mssg.metadata._src_node_id = task_ins.task.consumer.node_id
            # Convert to TaskRes
            task_res = message_to_taskres(out_mssg)
            # Store TaskRes in state
            state_factory.state().store_task_res(task_res)

        except Exception as ex:
            # TODO: gen TaskRes with relevant error, add it to state_factory.state()
            log(ERROR, ex)
            log(ERROR, traceback.format_exc())
            break


async def generate_pull_requests(
    queue: TaskInsQueue,
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
) -> None:
    """Generate TaskIns and add it to the queue."""
    while True:
        for node_id in nodes_mapping.keys():
            task_ins = state_factory.state().get_task_ins(node_id=node_id, limit=1)
            if task_ins:
                await queue.put(task_ins[0])
        log(INFO, f"TaskIns in queue: {queue.qsize()}")
        await asyncio.sleep(1.0)  # TODO: what's the right value here ?


async def run(
    app: Callable[[], ClientApp],
    backend: Backend,
    nodes_mapping: NodeToPartitionMapping,
    state_factory: StateFactory,
    node_states: Dict[int, NodeState],
) -> None:
    """Run the VCE async."""
    queue: TaskInsQueue = asyncio.Queue(64)  # TODO: how to set?

    # Build backend
    await backend.build()
    worker_tasks = [
        asyncio.create_task(
            worker(app, queue, node_states, state_factory, nodes_mapping, backend)
        )
        for _ in range(backend.num_workers)
    ]
    asyncio.create_task(generate_pull_requests(queue, state_factory, nodes_mapping))
    await queue.join()
    await asyncio.gather(*worker_tasks)


def run_vce(
    num_supernodes: int,
    client_resources: Dict[str, Union[float, int]],
    client_app_str: str,
    working_dir: str,
    state_factory: StateFactory,
    backend_str: str = "ray",
) -> None:
    """Run VirtualClientEnginge."""
    # Register nodes (as many as number of possible clients)
    # Each node has its own state
    node_states: Dict[int, NodeState] = {}
    nodes_mapping = _register_nodes(
        num_nodes=num_supernodes, state_factory=state_factory
    )
    for node_id in nodes_mapping.keys():
        node_states[node_id] = NodeState()

    try:
        backend_type: Type[RayBackend] = supported_backends[backend_str]
        backend = backend_type(client_resources, wdir=working_dir)
    except KeyError as ex:
        log(
            ERROR,
            f"Backennd type `{backend_str}`, is not supported."
            f" Use any of {list(supported_backends.keys())}",
        )
        raise (ex)

    log(INFO, f"Registered {len(nodes_mapping)} nodes")

    log(INFO, f"{client_app_str = }")

    def _load() -> ClientApp:
        app: ClientApp = load_client_app(client_app_str)
        return app

    app = _load

    asyncio.run(
        run(
            app,
            backend,
            nodes_mapping,
            state_factory,
            node_states,
        )
    )
