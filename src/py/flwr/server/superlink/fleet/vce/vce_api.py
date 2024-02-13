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

from logging import INFO
from time import sleep
from typing import Dict, List

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
from flwr.server.superlink.fleet.message_handler.message_handler import (
    PullTaskInsRequest,
    PushTaskResRequest,
    create_node,
    delete_node,
    pull_task_ins,
    push_task_res,
)
from flwr.server.superlink.state import StateFactory
from flwr.simulation.ray_transport.ray_actor import (
    DefaultActor,
    VirtualClientEngineActorPool,
)


def _construct_actor_pool() -> VirtualClientEngineActorPool:
    """Prepare ActorPool."""
    # TODO: these should be passed by the user (controls degree of parallelism)
    client_resources = {"num_cpus": 2, "num_gpus": 0.0}

    def _create_actor_fn():
        return DefaultActor.options(**client_resources).remote()

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
            task_type=taskins.task.task_type,
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


def run_vce(
    num_supernodes: int,
    client_app_callable_str: str,
    state_factory: StateFactory,
) -> None:
    """Run VirtualClientEnginge."""
    # Create actor pool
    pool = _construct_actor_pool()
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

    # Pull messages forever
    while True:
        sleep(3)
        # Pull task for each node
        for node in nodes:
            task_ins_pulled = pull_task_ins(
                request=PullTaskInsRequest(node=node), state=state_factory.state()
            )
            if task_ins_pulled.task_ins_list:
                print(f"Tasks PULLED for NODE {node}")
                node_id = node.node_id

                for task_ins in task_ins_pulled.task_ins_list:
                    # register and retrive runstate
                    node_states[node_id].register_context(run_id=task_ins.run_id)
                    run_state = node_states[node_id].retrieve_context(
                        run_id=task_ins.run_id
                    )

                    # convert TaskIns to Message
                    message = taskins_to_message(task_ins)

                    # Submite a task to the pool
                    pool.submit_client_job(
                        lambda a, a_fn, mssg, cid, state: a.run.remote(
                            a_fn, mssg, cid, state
                        ),
                        (app, message, str(node_id), run_state),
                    )

                    # Wait until result is ready
                    out_mssg, updated_runstate = pool.get_client_result(
                        str(node_id), timeout=None
                    )

                    # Update runstate
                    node_states[node_id].update_context(
                        task_ins.run_id, updated_runstate
                    )

                    # TODO: can we do the below in the VCE? this
                    # TODO: currently works because we run things sequentially
                    task_res = message_to_taskres(out_mssg)
                    task_res = configure_task_res(task_res, task_ins, node)
                    to_push = PushTaskResRequest(task_res_list=[task_res])
                    push_task_res(request=to_push, state=state_factory.state())

    # Delete nodes from state

    print("Deleting nodes...")
    for node in nodes:
        response = delete_node(node, state=state_factory.state())
        print(response)

    print("DONE")
    ray.shutdown()
