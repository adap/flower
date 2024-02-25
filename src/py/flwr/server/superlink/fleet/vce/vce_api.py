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
import json
import traceback
from logging import DEBUG, ERROR, INFO
from typing import Callable, Dict, Optional

from flwr.client.clientapp import ClientApp, load_client_app
from flwr.client.node_state import NodeState
from flwr.common.logger import log
from flwr.common.serde import message_from_taskins, message_to_taskres
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611
from flwr.server.superlink.state import StateFactory

from .backend import Backend, error_messages_backends, supported_backends

TaskInsQueue = asyncio.Queue[TaskIns]
NodeToPartitionMapping = Dict[int, int]


def _register_nodes(
    num_nodes: int, state_factory: StateFactory
) -> NodeToPartitionMapping:
    """Register nodes with the StateFactory and create node-id:partition-id mapping."""
    nodes_mapping: NodeToPartitionMapping = {}
    state = state_factory.state()
    for i in range(num_nodes):
        node_id = state.create_node()
        nodes_mapping[node_id] = i
    log(INFO, "Registered %i nodes", len(nodes_mapping))
    return nodes_mapping


# pylint: disable=too-many-arguments
async def worker(
    app: Callable[[], ClientApp],
    queue: TaskInsQueue,
    node_states: Dict[int, NodeState],
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
    backend: Backend,
) -> None:
    """Get TaskIns from queue and pass it to an actor in the pool to execute it."""
    state = state_factory.state()
    while True:
        try:
            task_ins = await queue.get()
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
                app, message, context
            )

            # Update Context
            node_states[node_id].update_context(
                task_ins.run_id, context=updated_context
            )

            # Undo change node_id for partition choice
            out_mssg.metadata._src_node_id = (  # pylint: disable=protected-access
                task_ins.task.consumer.node_id
            )
            # Convert to TaskRes
            task_res = message_to_taskres(out_mssg)
            # Store TaskRes in state
            state.store_task_res(task_res)

        except Exception as ex:  # pylint: disable=broad-exception-caught
            # pylint: disable=fixme
            # TODO: gen TaskRes with relevant error, add it to state_factory
            log(ERROR, ex)
            log(ERROR, traceback.format_exc())
            break


async def generate_pull_requests(
    queue: TaskInsQueue,
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
) -> None:
    """Generate TaskIns and add it to the queue."""
    state = state_factory.state()
    while True:
        for node_id in nodes_mapping.keys():
            task_ins = state.get_task_ins(node_id=node_id, limit=1)
            if task_ins:
                await queue.put(task_ins[0])
        log(DEBUG, "TaskIns in queue: %i", queue.qsize())
        # pylint: disable=fixme
        await asyncio.sleep(1.0)  # TODO: revisit


async def run(
    app: Callable[[], ClientApp],
    backend: Backend,
    nodes_mapping: NodeToPartitionMapping,
    state_factory: StateFactory,
    node_states: Dict[int, NodeState],
) -> None:
    """Run the VCE async."""
    # pylint: disable=fixme
    queue: TaskInsQueue = asyncio.Queue(64)  # TODO: revisit

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


# pylint: disable=too-many-arguments,unused-argument
def start_vce(
    num_supernodes: int,
    client_app_module_name: str,
    backend_name: str,
    backend_config_json_stream: str,
    state_factory: StateFactory,
    working_dir: str,
    f_stop: Optional[asyncio.Event] = None,
) -> None:
    """Start Fleet API with the VirtualClientEngine (VCE)."""
    # Register SuperNodes
    nodes_mapping = _register_nodes(
        num_nodes=num_supernodes, state_factory=state_factory
    )

    # Construct mapping of NodeStates
    node_states: Dict[int, NodeState] = {}
    for node_id in nodes_mapping:
        node_states[node_id] = NodeState()

    # Load backend config
    log(INFO, "Supported backends: %s", list(supported_backends.keys()))
    backend_config = json.loads(backend_config_json_stream)

    try:
        backend_type = supported_backends[backend_name]
        backend = backend_type(backend_config, work_dir=working_dir)
    except KeyError as ex:
        log(
            ERROR,
            "Backend `%s`, is not supported. Use any of %s or add support "
            "for a new backend.",
            backend_name,
            list(supported_backends.keys()),
        )
        if backend_name in error_messages_backends:
            log(ERROR, error_messages_backends[backend_name])

        raise ex

    log(INFO, "client_app_module_name = %s", client_app_module_name)

    def _load() -> ClientApp:
        app: ClientApp = load_client_app(client_app_module_name)
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
