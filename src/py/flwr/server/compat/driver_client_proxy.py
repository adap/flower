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
from typing import List, Optional

from flwr import common
from flwr.common import recordset_compat as compat
from flwr.common import serde
from flwr.common.constant import (
    MESSAGE_TYPE_EVALUATE,
    MESSAGE_TYPE_FIT,
    MESSAGE_TYPE_GET_PARAMETERS,
    MESSAGE_TYPE_GET_PROPERTIES,
)
from flwr.common.recordset import RecordSet
from flwr.proto import driver_pb2, node_pb2, task_pb2  # pylint: disable=E0611
from flwr.server.client_proxy import ClientProxy

from ..driver.grpc_driver import GrpcDriver

SLEEP_TIME = 1


class DriverClientProxy(ClientProxy):
    """Flower client proxy which delegates work using the Driver API."""

    def __init__(self, node_id: int, driver: GrpcDriver, anonymous: bool, run_id: int):
        super().__init__(str(node_id))
        self.node_id = node_id
        self.driver = driver
        self.run_id = run_id
        self.anonymous = anonymous

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        # Ins to RecordSet
        out_recordset = compat.getpropertiesins_to_recordset(ins)
        # Fetch response
        in_recordset = self._send_receive_recordset(
            out_recordset, MESSAGE_TYPE_GET_PROPERTIES, timeout
        )
        # RecordSet to Res
        return compat.recordset_to_getpropertiesres(in_recordset)

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        # Ins to RecordSet
        out_recordset = compat.getparametersins_to_recordset(ins)
        # Fetch response
        in_recordset = self._send_receive_recordset(
            out_recordset, MESSAGE_TYPE_GET_PARAMETERS, timeout
        )
        # RecordSet to Res
        return compat.recordset_to_getparametersres(in_recordset, False)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        # Ins to RecordSet
        out_recordset = compat.fitins_to_recordset(ins, keep_input=True)
        # Fetch response
        in_recordset = self._send_receive_recordset(
            out_recordset, MESSAGE_TYPE_FIT, timeout
        )
        # RecordSet to Res
        return compat.recordset_to_fitres(in_recordset, keep_input=False)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        # Ins to RecordSet
        out_recordset = compat.evaluateins_to_recordset(ins, keep_input=True)
        # Fetch response
        in_recordset = self._send_receive_recordset(
            out_recordset, MESSAGE_TYPE_EVALUATE, timeout
        )
        # RecordSet to Res
        return compat.recordset_to_evaluateres(in_recordset)

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)

    def _send_receive_recordset(
        self,
        recordset: RecordSet,
        task_type: str,
        timeout: Optional[float],
    ) -> RecordSet:
        task_ins = task_pb2.TaskIns(  # pylint: disable=E1101
            task_id="",
            group_id="",
            run_id=self.run_id,
            task=task_pb2.Task(  # pylint: disable=E1101
                producer=node_pb2.Node(  # pylint: disable=E1101
                    node_id=0,
                    anonymous=True,
                ),
                consumer=node_pb2.Node(  # pylint: disable=E1101
                    node_id=self.node_id,
                    anonymous=self.anonymous,
                ),
                task_type=task_type,
                recordset=serde.recordset_to_proto(recordset),
            ),
        )
        push_task_ins_req = driver_pb2.PushTaskInsRequest(  # pylint: disable=E1101
            task_ins_list=[task_ins]
        )

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
            pull_task_res_req = driver_pb2.PullTaskResRequest(  # pylint: disable=E1101
                node=node_pb2.Node(node_id=0, anonymous=True),  # pylint: disable=E1101
                task_ids=[task_id],
            )

            # Ask Driver API for TaskRes
            pull_task_res_res = self.driver.pull_task_res(req=pull_task_res_req)

            task_res_list: List[task_pb2.TaskRes] = list(  # pylint: disable=E1101
                pull_task_res_res.task_res_list
            )
            if len(task_res_list) == 1:
                task_res = task_res_list[0]
                return serde.recordset_from_proto(task_res.task.recordset)

            if timeout is not None and time.time() > start_time + timeout:
                raise RuntimeError("Timeout reached")
            time.sleep(SLEEP_TIME)
