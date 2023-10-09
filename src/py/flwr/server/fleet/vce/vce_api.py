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
from typing import cast

import ray

from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.message_handler.task_handler import (
    configure_task_res,
    get_server_message_from_task_ins,
    wrap_client_message_in_task_res,
)
from flwr.common import EvaluateRes, FitRes, GetParametersRes, GetPropertiesRes, serde
from flwr.common.logger import log
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
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


def _get_job_fn_and_response_conversion_for_legacy_message(server_msg: ServerMessage):
    """Return job clients should execute given the message and how to prepare result."""
    field = server_msg.WhichOneof("msg")

    if field == "get_properties_ins":

        def func(client):
            return maybe_call_get_properties(
                client=client,
                get_properties_ins=server_msg.get_properties_ins,
            )

        def post_func(get_properties_res):
            return ClientMessage(
                get_properties_res=serde.get_properties_res_to_proto(
                    cast(GetPropertiesRes, get_properties_res)
                )
            )

    elif field == "get_parameters_ins":

        def func(client):
            return maybe_call_get_parameters(
                client=client,
                get_parameters_ins=server_msg.get_parameters_ins,
            )

        def post_func(get_params_res):
            return ClientMessage(
                get_parameters_res=serde.get_parameters_res_to_proto(
                    cast(GetParametersRes, get_params_res)
                )
            )

    elif field == "fit_ins":

        def func(client):
            return maybe_call_fit(
                client=client,
                fit_ins=server_msg.fit_ins,
            )

        def post_func(fit_res):
            return ClientMessage(fit_res=serde.fit_res_to_proto(cast(FitRes, fit_res)))

    elif field == "evaluate_ins":

        def func(client):
            return maybe_call_evaluate(
                client=client,
                evaluate_ins=server_msg.evaluate_ins,
            )

        def post_func(eval_res):
            return ClientMessage(
                evaluate_res=serde.evaluate_res_to_proto(cast(EvaluateRes, eval_res))
            )

    else:
        raise NotImplementedError(f"Message with field '{field}' not understood.")

    return func, post_func


def run_vce(num_clients: int, state_factory: StateFactory):
    """Run VirtualClientEnginge."""
    # Create actor pool
    pool = _construct_actor_pool()
    log(INFO, f"Constructed ActorPool with: {pool.num_actors} actors")

    # Register nodes (as many as number of possible clients)
    create_node_responses = []

    for _ in range(num_clients):
        res = create_node(request=None, state=state_factory.state())
        create_node_responses.append(res)

    log(INFO, f"Registered {len(create_node_responses)} nodes")

    # Pull messages forever
    while True:
        sleep(3)
        # Pull task for each node
        for res in create_node_responses:
            task_ins_pulled = pull_task_ins(request=res, state=state_factory.state())
            if task_ins_pulled.task_ins_list:
                print(f"Tasks PULLED for NODE {res.node}")

                for task_ins in task_ins_pulled.task_ins_list:
                    # get message field
                    server_msg = get_server_message_from_task_ins(
                        task_ins, exclude_reconnect_ins=False
                    )
                    if server_msg is None:
                        raise NotImplementedError("Can only handle legacy messages...")

                    # Determine how to process message and prepare response
                    (
                        func,
                        post_func,
                    ) = _get_job_fn_and_response_conversion_for_legacy_message(
                        server_msg
                    )

                    # Submite a task to the pool
                    pool.submit_client_job(
                        lambda a, c_fn, j_fn, cid_: a.run.remote(c_fn, j_fn, cid_),
                        (client_fn, func, res.node.node_id),
                    )

                    # Wait until result is ready
                    result = pool.get_client_result(res.node.node_id, timeout=None)

                    client_message = post_func(result)
                    task_res = wrap_client_message_in_task_res(client_message)
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
from flwr.client import NumPyClient
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
NUM_TRAIN=512
NUM_TEST=256

# Define a dummy client
class DummyFlowerClient(NumPyClient):
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"],)
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
    return DummyFlowerClient()