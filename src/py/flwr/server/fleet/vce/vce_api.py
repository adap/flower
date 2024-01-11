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

# construct nodes
from typing import Dict

import ray

from flwr.client.message_handler.task_handler import configure_task_res
from flwr.client.node_state import NodeState
from flwr.common.logger import log
from flwr.server.fleet.message_handler.message_handler import (
    PushTaskResRequest,
    create_node,
    delete_node,
    pull_task_ins,
    push_task_res,
)
from flwr.server.state import StateFactory


def _construct_actor_pool():
    """Prepare ActorPool."""
    # TODO: imports are here due to hard-to-fix circular import
    from flwr.simulation.ray_transport.ray_actor import (
        DefaultActor,
        VirtualClientEngineActorPool,
    )

    client_resources = {"num_cpus": 2, "num_gpus": 0.0}

    def create_actor_fn():
        return DefaultActor.options(**client_resources).remote()

    # Create actor pool
    ray.init(include_dashboard=True)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )
    return pool


def run_vce(num_clients: int, state_factory: StateFactory):
    """Run VirtualClientEnginge."""
    # Create actor pool
    pool = _construct_actor_pool()
    log(INFO, f"Constructed ActorPool with: {pool.num_actors} actors")

    # Register nodes (as many as number of possible clients)
    # Each node has its own state
    node_states: Dict[int, NodeState] = {}
    create_node_responses = []
    for _ in range(num_clients):
        res = create_node(request=None, state=state_factory.state())
        create_node_responses.append(res)
        node_states[res.node.node_id] = NodeState()

    log(INFO, f"Registered {len(create_node_responses)} nodes")

    # Pull messages forever
    while True:
        sleep(3)
        # Pull task for each node
        for res in create_node_responses:
            task_ins_pulled = pull_task_ins(request=res, state=state_factory.state())
            if task_ins_pulled.task_ins_list:
                print(f"Tasks PULLED for NODE {res.node}")
                node_id = res.node.node_id

                for task_ins in task_ins_pulled.task_ins_list:
                    # register and retrive runstate
                    node_states[node_id].register_runstate(run_id=task_ins.run_id)
                    run_state = node_states[node_id].retrieve_runstate(
                        run_id=task_ins.run_id
                    )

                    # Submite a task to the pool
                    pool.submit_task_ins(
                        lambda a, c_fn, t_ins, cid, state: a.run.remote(
                            c_fn, t_ins, cid, state
                        ),
                        (client_fn, task_ins, node_id, run_state),
                    )

                    # Wait until result is ready
                    task_res, updated_runstate = pool.get_client_result(
                        node_id, timeout=None
                    )

                    # Update runstate
                    node_states[node_id].update_runstate(
                        task_ins.run_id, updated_runstate
                    )

                    # TODO: can we do the below in the VCE? this currently works because we run things sequentially
                    task_res = configure_task_res(task_res, task_ins, res.node)
                    to_push = PushTaskResRequest(task_res_list=[task_res])
                    push_task_res(request=to_push, state=state_factory.state())

    # Delete nodes from state

    print("Deleting nodes...")
    for res in create_node_responses:
        response = delete_node(res, state=state_factory.state())
        print(response)

    print("DONE")
    ray.shutdown()


### Dummy code providing a Client and client_fn
from random import random

import tensorflow as tf

from flwr.client import NumPyClient

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
NUM_TRAIN = 512
NUM_TEST = 256


# Define a dummy client
class DummyFlowerClient(NumPyClient):
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), classes=10, weights=None
        )
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            x_train[:NUM_TRAIN],
            y_train[:NUM_TRAIN],
            batch_size=32,
            epochs=1,
            validation_split=0.1,
        )
        results = {
            "train_loss": history.history["loss"][0],
            "train_accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return self.model.get_weights(), len(x_train[:NUM_TRAIN]), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test[:NUM_TEST], y_test[:NUM_TEST])
        return random(), len(x_test[:NUM_TEST]), {"accuracy": accuracy}


def client_fn(cid: str = None):
    return DummyFlowerClient().to_client()
