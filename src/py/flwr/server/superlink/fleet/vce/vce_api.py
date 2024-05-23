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
"""Fleet Simulation Engine API."""

import asyncio
import json
import sys
import time
import traceback
from logging import DEBUG, ERROR, INFO, WARN
from typing import Callable, Dict, List, Optional

from flwr.client.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.client.node_state import NodeState
from flwr.common.constant import PING_MAX_INTERVAL, ErrorCode
from flwr.common.logger import log
from flwr.common.message import Error
from flwr.common.object_ref import load_app
from flwr.common.serde import message_from_taskins, message_to_taskres
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611
from flwr.server.superlink.state import StateFactory

from .backend import Backend, error_messages_backends, supported_backends

NodeToPartitionMapping = Dict[int, int]


def _register_nodes(
    num_nodes: int, state_factory: StateFactory
) -> NodeToPartitionMapping:
    """Register nodes with the StateFactory and create node-id:partition-id mapping."""
    nodes_mapping: NodeToPartitionMapping = {}
    state = state_factory.state()
    for i in range(num_nodes):
        node_id = state.create_node(ping_interval=PING_MAX_INTERVAL)
        nodes_mapping[node_id] = i
    log(DEBUG, "Registered %i nodes", len(nodes_mapping))
    return nodes_mapping


# pylint: disable=too-many-arguments,too-many-locals
async def worker(
    app_fn: Callable[[], ClientApp],
    queue: "asyncio.Queue[TaskIns]",
    node_states: Dict[int, NodeState],
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
    backend: Backend,
) -> None:
    """Get TaskIns from queue and pass it to an actor in the pool to execute it."""
    state = state_factory.state()
    while True:
        out_mssg = None
        try:
            task_ins: TaskIns = await queue.get()
            node_id = task_ins.task.consumer.node_id

            # Register and retrieve runstate
            node_states[node_id].register_context(run_id=task_ins.run_id)
            context = node_states[node_id].retrieve_context(run_id=task_ins.run_id)

            # Convert TaskIns to Message
            message = message_from_taskins(task_ins)
            # Set partition_id
            message.metadata.partition_id = nodes_mapping[node_id]

            # Let backend process message
            out_mssg, updated_context = await backend.process_message(
                app_fn, message, context
            )

            # Update Context
            node_states[node_id].update_context(
                task_ins.run_id, context=updated_context
            )

        except asyncio.CancelledError as e:
            log(DEBUG, "Terminating async worker: %s", e)
            break

        # Exceptions aren't raised but reported as an error message
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, ex)
            log(ERROR, traceback.format_exc())

            if isinstance(ex, ClientAppException):
                e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
            elif isinstance(ex, LoadClientAppError):
                e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
            else:
                e_code = ErrorCode.UNKNOWN

            reason = str(type(ex)) + ":<'" + str(ex) + "'>"
            out_mssg = message.create_error_reply(
                error=Error(code=e_code, reason=reason)
            )

        finally:
            if out_mssg:
                # Convert to TaskRes
                task_res = message_to_taskres(out_mssg)
                # Store TaskRes in state
                task_res.task.pushed_at = time.time()
                state.store_task_res(task_res)


async def add_taskins_to_queue(
    queue: "asyncio.Queue[TaskIns]",
    state_factory: StateFactory,
    nodes_mapping: NodeToPartitionMapping,
    backend: Backend,
    consumers: List["asyncio.Task[None]"],
    f_stop: asyncio.Event,
) -> None:
    """Retrieve TaskIns and add it to the queue."""
    state = state_factory.state()
    num_initial_consumers = len(consumers)
    while not f_stop.is_set():
        for node_id in nodes_mapping.keys():
            task_ins = state.get_task_ins(node_id=node_id, limit=1)
            if task_ins:
                await queue.put(task_ins[0])

        # Count consumers that are running
        num_active = sum(not (cc.done()) for cc in consumers)

        # Alert if number of consumers decreased by half
        if num_active < num_initial_consumers // 2:
            log(
                WARN,
                "Number of active workers has more than halved: (%i/%i active)",
                num_active,
                num_initial_consumers,
            )

        # Break if consumers died
        if num_active == 0:
            raise RuntimeError("All workers have died. Ending Simulation.")

        # Log some stats
        log(
            DEBUG,
            "Simulation Engine stats: "
            "Active workers: (%i/%i) | %s (%i workers) | Tasks in queue: %i)",
            num_active,
            num_initial_consumers,
            backend.__class__.__name__,
            backend.num_workers,
            queue.qsize(),
        )
        await asyncio.sleep(1.0)
    log(DEBUG, "Async producer: Stopped pulling from StateFactory.")


async def run(
    app_fn: Callable[[], ClientApp],
    backend_fn: Callable[[], Backend],
    nodes_mapping: NodeToPartitionMapping,
    state_factory: StateFactory,
    node_states: Dict[int, NodeState],
    f_stop: asyncio.Event,
) -> None:
    """Run the VCE async."""
    queue = asyncio.Queue(128)

    try:

        # Instantiate backend
        backend = backend_fn()

        # Build backend
        await backend.build()

        # Add workers (they submit Messages to Backend)
        worker_tasks = [
            asyncio.create_task(
                worker(
                    app_fn, queue, node_states, state_factory, nodes_mapping, backend
                )
            )
            for _ in range(backend.num_workers)
        ]
        # Create producer (adds TaskIns into Queue)
        producer = asyncio.create_task(
            add_taskins_to_queue(
                queue, state_factory, nodes_mapping, backend, worker_tasks, f_stop
            )
        )

        # Wait for producer to finish
        # The producer runs forever until f_stop is set or until
        # all worker (consumer) coroutines are completed. Workers
        # also run forever and only end if an exception is raised.
        await asyncio.gather(producer)

    except Exception as ex:

        log(ERROR, "An exception occured!! %s", ex)
        log(ERROR, traceback.format_exc())
        log(WARN, "Stopping Simulation Engine.")

        # Manually trigger stopping event
        f_stop.set()

        # Raise exception
        raise RuntimeError("Simulation Engine crashed.") from ex

    finally:
        # Produced task terminated, now cancel worker tasks
        for w_t in worker_tasks:
            _ = w_t.cancel()

        while not all(w_t.done() for w_t in worker_tasks):
            log(DEBUG, "Terminating async workers...")
            await asyncio.sleep(0.5)

        await asyncio.gather(*[w_t for w_t in worker_tasks if not w_t.done()])

        # Terminate backend
        await backend.terminate()


# pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# pylint: disable=too-many-statements
def start_vce(
    backend_name: str,
    backend_config_json_stream: str,
    app_dir: str,
    f_stop: asyncio.Event,
    client_app: Optional[ClientApp] = None,
    client_app_attr: Optional[str] = None,
    num_supernodes: Optional[int] = None,
    state_factory: Optional[StateFactory] = None,
    existing_nodes_mapping: Optional[NodeToPartitionMapping] = None,
) -> None:
    """Start Fleet API with the Simulation Engine."""
    if client_app_attr is not None and client_app is not None:
        raise ValueError(
            "Both `client_app_attr` and `client_app` are provided, "
            "but only one is allowed."
        )

    if num_supernodes is not None and existing_nodes_mapping is not None:
        raise ValueError(
            "Both `num_supernodes` and `existing_nodes_mapping` are provided, "
            "but only one is allowed."
        )
    if num_supernodes is None:
        if state_factory is None or existing_nodes_mapping is None:
            raise ValueError(
                "If not passing an existing `state_factory` and associated "
                "`existing_nodes_mapping` you must supply `num_supernodes` to indicate "
                "how many nodes to insert into a new StateFactory that will be created."
            )
    if existing_nodes_mapping:
        if state_factory is None:
            raise ValueError(
                "`existing_nodes_mapping` was passed, but no `state_factory` was "
                "passed."
            )
        log(INFO, "Using exiting NodeToPartitionMapping and StateFactory.")
        # Use mapping constructed externally. This also means nodes
        # have previously being registered.
        nodes_mapping = existing_nodes_mapping

    if not state_factory:
        log(INFO, "A StateFactory was not supplied to the SimulationEngine.")
        # Create an empty in-memory state factory
        state_factory = StateFactory(":flwr-in-memory-state:")
        log(INFO, "Created new %s.", state_factory.__class__.__name__)

    if num_supernodes:
        # Register SuperNodes
        nodes_mapping = _register_nodes(
            num_nodes=num_supernodes, state_factory=state_factory
        )

    # Construct mapping of NodeStates
    node_states: Dict[int, NodeState] = {}
    for node_id in nodes_mapping:
        node_states[node_id] = NodeState()

    # Load backend config
    log(DEBUG, "Supported backends: %s", list(supported_backends.keys()))
    backend_config = json.loads(backend_config_json_stream)

    try:
        backend_type = supported_backends[backend_name]
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

    def backend_fn() -> Backend:
        """Instantiate a Backend."""
        return backend_type(backend_config, work_dir=app_dir)

    # Load ClientApp if needed
    def _load() -> ClientApp:

        if client_app_attr:

            if app_dir is not None:
                sys.path.insert(0, app_dir)

            app: ClientApp = load_app(client_app_attr, LoadClientAppError)

            if not isinstance(app, ClientApp):
                raise LoadClientAppError(
                    f"Attribute {client_app_attr} is not of type {ClientApp}",
                ) from None

        if client_app:
            app = client_app
        return app

    app_fn = _load

    try:
        # Test if ClientApp can be loaded
        _ = app_fn()

        # Run main simulation loop
        asyncio.run(
            run(
                app_fn,
                backend_fn,
                nodes_mapping,
                state_factory,
                node_states,
                f_stop,
            )
        )
    except LoadClientAppError as loadapp_ex:
        f_stop_delay = 10
        log(
            ERROR,
            "LoadClientAppError exception encountered. Terminating simulation in %is",
            f_stop_delay,
        )
        time.sleep(f_stop_delay)
        f_stop.set()  # set termination event
        raise loadapp_ex
    except Exception as ex:
        raise ex
