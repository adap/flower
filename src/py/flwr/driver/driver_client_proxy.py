# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower ClientProxy implementation for Driver API."""


import time
from typing import List, Optional, cast

from flwr import common
from flwr.common import serde
from flwr.proto import driver_pb2, node_pb2, task_pb2, transport_pb2
from flwr.server.client_proxy import ClientProxy

from .driver import Driver

SLEEP_TIME = 1


class DriverClientProxy(ClientProxy):
    """Flower client proxy which delegates work using the Driver API."""

    def __init__(self, node_id: int, driver: Driver, anonymous: bool, workload_id: int):
        super().__init__(str(node_id))
        self.node_id = node_id
        self.driver = driver
        self.workload_id = workload_id
        self.anonymous = anonymous

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        server_message_proto: transport_pb2.ServerMessage = (
            serde.server_message_to_proto(
                server_message=common.ServerMessage(get_properties_ins=ins)
            )
        )
        return cast(
            common.GetPropertiesRes,
            self._send_receive_msg(server_message_proto, timeout).get_properties_res,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        server_message_proto: transport_pb2.ServerMessage = (
            serde.server_message_to_proto(
                server_message=common.ServerMessage(get_parameters_ins=ins)
            )
        )
        return cast(
            common.GetParametersRes,
            self._send_receive_msg(server_message_proto, timeout).get_parameters_res,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        server_message_proto: transport_pb2.ServerMessage = (
            serde.server_message_to_proto(
                server_message=common.ServerMessage(fit_ins=ins)
            )
        )
        return cast(
            common.FitRes,
            self._send_receive_msg(server_message_proto, timeout).fit_res,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        server_message_proto: transport_pb2.ServerMessage = (
            serde.server_message_to_proto(
                server_message=common.ServerMessage(evaluate_ins=ins)
            )
        )
        return cast(
            common.EvaluateRes,
            self._send_receive_msg(server_message_proto, timeout).evaluate_res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)

    def _send_receive_msg(
        self, server_message: transport_pb2.ServerMessage, timeout: Optional[float]
    ) -> transport_pb2.ClientMessage:
        task_ins = task_pb2.TaskIns(
            task_id="",
            group_id="",
            workload_id=self.workload_id,
            task=task_pb2.Task(
                producer=node_pb2.Node(
                    node_id=0,
                    anonymous=True,
                ),
                consumer=node_pb2.Node(
                    node_id=self.node_id,
                    anonymous=self.anonymous,
                ),
                legacy_server_message=server_message,
            ),
        )
        push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_list=[task_ins])

        # Send TaskIns to Driver API
        push_task_ins_res = self.driver.push_task_ins(req=push_task_ins_req)

        if len(push_task_ins_res.task_ids) != 1:
            raise ValueError("Unexpected number of task_ids")

        task_id = push_task_ins_res.task_ids[0]
        if task_id == "":
            raise ValueError(f"Failed to schedule task for node {self.node_id}")

        if timeout:
            start_time = time.time()

        while True:
            pull_task_res_req = driver_pb2.PullTaskResRequest(
                node=node_pb2.Node(node_id=0, anonymous=True),
                task_ids=[task_id],
            )

            # Ask Driver API for TaskRes
            pull_task_res_res = self.driver.pull_task_res(req=pull_task_res_req)

            task_res_list: List[task_pb2.TaskRes] = list(
                pull_task_res_res.task_res_list
            )
            if len(task_res_list) == 1:
                task_res = task_res_list[0]
                return serde.client_message_from_proto(  # type: ignore
                    task_res.task.legacy_client_message
                )

            if timeout is not None and time.time() > start_time + timeout:
                raise RuntimeError("Timeout reached")
            time.sleep(SLEEP_TIME)
