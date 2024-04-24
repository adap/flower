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
from typing import Optional

from flwr import common
from flwr.common import DEFAULT_TTL, Message, MessageType, MessageTypeLegacy, RecordSet
from flwr.common import recordset_compat as compat
from flwr.server.client_proxy import ClientProxy

from ..driver.driver import Driver

SLEEP_TIME = 1


class DriverClientProxy(ClientProxy):
    """Flower client proxy which delegates work using the Driver API."""

    def __init__(self, node_id: int, driver: Driver, anonymous: bool, run_id: int):
        super().__init__(str(node_id))
        self.node_id = node_id
        self.driver = driver
        self.run_id = run_id
        self.anonymous = anonymous

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
        in_recordset = self._send_receive_recordset(
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
        in_recordset = self._send_receive_recordset(
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
        in_recordset = self._send_receive_recordset(
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
        in_recordset = self._send_receive_recordset(
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

    def _send_receive_recordset(
        self,
        recordset: RecordSet,
        task_type: str,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> RecordSet:

        # Create message
        message = self.driver.create_message(
            content=recordset,
            message_type=task_type,
            dst_node_id=self.node_id,
            group_id=str(group_id) if group_id else "",
            ttl=DEFAULT_TTL,
        )

        # Push message
        message_ids = list(self.driver.push_messages(messages=[message]))
        if len(message_ids) != 1:
            raise ValueError("Unexpected number of message_ids")

        message_id = message_ids[0]
        if message_id == "":
            raise ValueError(f"Failed to send message to node {self.node_id}")

        if timeout:
            start_time = time.time()

        while True:
            messages = list(self.driver.pull_messages(message_ids))
            if len(messages) == 1:
                msg: Message = messages[0]
                return msg.content

            if timeout is not None and time.time() > start_time + timeout:
                raise RuntimeError("Timeout reached")
            time.sleep(SLEEP_TIME)
