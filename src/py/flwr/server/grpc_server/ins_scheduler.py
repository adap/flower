# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Instruction scheduler for the legacy gRPC transport stack."""


import threading
import time
from logging import DEBUG, ERROR
from typing import Dict, List, Optional

from flwr.common import EvaluateRes, FitRes, GetParametersRes, GetPropertiesRes, serde
from flwr.common.logger import log
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.state import State


class InsScheduler:
    """Schedule ClientProxy calls on a background thread."""

    def __init__(self, client_proxy: ClientProxy, state: State):
        self.client_proxy = client_proxy
        self.state = state
        self.worker_thread: Optional[threading.Thread] = None
        self.shared_memory_state = {"stop": False}

    def start(self) -> None:
        """Start the worker thread."""
        self.worker_thread = threading.Thread(
            target=_worker,
            args=(
                self.client_proxy,
                self.shared_memory_state,
                self.state,
            ),
        )
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        if self.worker_thread is None:
            log(ERROR, "InsScheduler.stop called, but worker_thread is None")
            return
        self.shared_memory_state["stop"] = True
        self.worker_thread.join()
        self.worker_thread = None
        self.shared_memory_state["stop"] = False


def _worker(
    client_proxy: ClientProxy,
    shared_memory_state: Dict[str, bool],
    state: State,
) -> None:
    """Sequentially call ClientProxy methods to process outstanding tasks."""
    log(DEBUG, "Worker for node %i started", client_proxy.node_id)
    while not shared_memory_state["stop"]:
        log(DEBUG, "Worker for node %i checking state", client_proxy.node_id)

        # Step 1: pull *Ins (next task) out of `state`
        task_ins_list: List[TaskIns] = state.get_task_ins(
            node_id=client_proxy.node_id,
            limit=1,
        )
        if not task_ins_list:
            log(DEBUG, "Worker for node %i: no task found", client_proxy.node_id)
            time.sleep(3)
            continue

        task_ins = task_ins_list[0]
        log(
            DEBUG,
            "Worker for node %i: FOUND task %s",
            client_proxy.node_id,
            task_ins.task_id,
        )

        # Step 2: call client_proxy.{fit,evaluate,...}
        server_message = task_ins.task.legacy_server_message
        client_message_proto = _call_client_proxy(
            client_proxy=client_proxy,
            server_message=server_message,
            timeout=None,
        )

        # Step 3: wrap FitRes in a ServerMessage in a Task in a TaskRes
        task_res = TaskRes(
            task_id="",  # Will be created and set by the State
            group_id="",
            workload_id="",
            task=Task(
                producer=Node(node_id=client_proxy.node_id, anonymous=False),
                legacy_client_message=client_message_proto,
                ancestry=[task_ins.task_id],
            ),
        )

        # Step 4: write *Res (result) back to `state`
        state.store_task_res(task_res=task_res)

    # Exit worker thread
    log(DEBUG, "Worker for node %i stopped", client_proxy.node_id)


def _call_client_proxy(
    client_proxy: ClientProxy, server_message: ServerMessage, timeout: Optional[float]
) -> ClientMessage:
    """."""

    # pylint: disable=too-many-locals

    field = server_message.WhichOneof("msg")

    if field == "get_properties_ins":
        get_properties_ins = serde.get_properties_ins_from_proto(
            msg=server_message.get_properties_ins
        )
        get_properties_res: GetPropertiesRes = client_proxy.get_properties(
            ins=get_properties_ins,
            timeout=timeout,
        )
        get_properties_res_proto = serde.get_properties_res_to_proto(
            res=get_properties_res
        )
        return ClientMessage(get_properties_res=get_properties_res_proto)

    if field == "get_parameters_ins":
        get_parameters_ins = serde.get_parameters_ins_from_proto(
            msg=server_message.get_parameters_ins
        )
        get_parameters_res: GetParametersRes = client_proxy.get_parameters(
            ins=get_parameters_ins,
            timeout=timeout,
        )
        get_parameters_res_proto = serde.get_parameters_res_to_proto(
            res=get_parameters_res
        )
        return ClientMessage(get_parameters_res=get_parameters_res_proto)

    if field == "fit_ins":
        fit_ins = serde.fit_ins_from_proto(msg=server_message.fit_ins)
        fit_res: FitRes = client_proxy.fit(
            ins=fit_ins,
            timeout=timeout,
        )
        fit_res_proto = serde.fit_res_to_proto(res=fit_res)
        return ClientMessage(fit_res=fit_res_proto)

    if field == "evaluate_ins":
        evaluate_ins = serde.evaluate_ins_from_proto(msg=server_message.evaluate_ins)
        evaluate_res: EvaluateRes = client_proxy.evaluate(
            ins=evaluate_ins,
            timeout=timeout,
        )
        evaluate_res_proto = serde.evaluate_res_to_proto(res=evaluate_res)
        return ClientMessage(evaluate_res=evaluate_res_proto)

    raise Exception(
        "Unsupported instruction in ServerMessage, cannot deserialize from ProtoBuf"
    )
