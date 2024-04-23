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


from typing import Optional

from flwr import common
from flwr.common import MessageType, MessageTypeLegacy, RecordSet
from flwr.common import recordset_compat as compat
from flwr.proto import task_pb2  # pylint: disable=E0611
from flwr.server.client_proxy import ClientProxy

SLEEP_TIME = 1


class DriverClientProxy(ClientProxy):
    """Generic Flower client proxy which delegates work using the Driver API."""

    def __init__(self, node_id: int, anonymous: bool, run_id: int):
        super().__init__(str(node_id))
        self.node_id = node_id
        self.run_id = run_id
        self.anonymous = anonymous

    def _send_receive_recordset(
        self,
        recordset: RecordSet,
        task_type: str,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> RecordSet:
        _ = (recordset, task_type, timeout, group_id)
        raise NotImplementedError(
            f"Use a {self.__class__.__name__} that implements this method."
        )

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        # Ins to RecordSet
        out_recordset = compat.getpropertiesins_to_recordset(ins)
        # Fetch response
        in_recordset = self._send_receive_recordset(  # pylint: disable=E1111
            out_recordset, MessageTypeLegacy.GET_PROPERTIES, timeout, group_id
        )
        # RecordSet to Res
        return compat.recordset_to_getpropertiesres(in_recordset)

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        # Ins to RecordSet
        out_recordset = compat.getparametersins_to_recordset(ins)
        # Fetch response
        in_recordset = self._send_receive_recordset(  # pylint: disable=E1111
            out_recordset, MessageTypeLegacy.GET_PARAMETERS, timeout, group_id
        )
        # RecordSet to Res
        return compat.recordset_to_getparametersres(in_recordset, False)

    def fit(
        self, ins: common.FitIns, timeout: Optional[float], group_id: Optional[int]
    ) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        # Ins to RecordSet
        out_recordset = compat.fitins_to_recordset(ins, keep_input=True)
        # Fetch response
        in_recordset = self._send_receive_recordset(  # pylint: disable=E1111
            out_recordset, MessageType.TRAIN, timeout, group_id
        )
        # RecordSet to Res
        return compat.recordset_to_fitres(in_recordset, keep_input=False)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float], group_id: Optional[int]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        # Ins to RecordSet
        out_recordset = compat.evaluateins_to_recordset(ins, keep_input=True)
        # Fetch response
        in_recordset = self._send_receive_recordset(  # pylint: disable=E1111
            out_recordset, MessageType.EVALUATE, timeout, group_id
        )
        # RecordSet to Res
        return compat.recordset_to_evaluateres(in_recordset)

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)


def validate_task_res(
    task_res: task_pb2.TaskRes,  # pylint: disable=E1101
) -> None:
    """Validate if a TaskRes is empty or not."""
    if not task_res.HasField("task"):
        raise ValueError("Invalid TaskRes, field `task` missing")
    if task_res.task.HasField("error"):
        raise ValueError("Exception during client-side task execution")
    if not task_res.task.HasField("recordset"):
        raise ValueError("Invalid TaskRes, both `recordset` and `error` are missing")
