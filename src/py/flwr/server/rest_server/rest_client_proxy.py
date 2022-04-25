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
"""REST-based Flower ClientProxy implementation."""


from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.state import State


class RestClientProxy(ClientProxy):
    """Flower client proxy which delegates over the network using REST.

    This ClientProxy implementation uses REST-related naming because it
    is used for the REST-based communication stack, but the
    implementation is not REST-specific. The approach of using a
    TaskManager to announce client tasks and gather task results can be
    applied to other commuication stack such as gRPC (for both
    request/response or streaming).
    """

    def __init__(
        self,
        cid: str,
    ):
        super().__init__(cid)

    def get_properties(self, ins: common.PropertiesIns) -> common.PropertiesRes:
        """Requests client's set of internal properties."""
        properties_msg = serde.properties_ins_to_proto(ins)
        server_msg = ServerMessage(properties_ins=properties_msg)
        client_msg = request(cid=self.cid, server_msg=server_msg)
        return serde.properties_res_from_proto(client_msg.properties_res)

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_to_proto()
        server_msg = ServerMessage(get_parameters=get_parameters_msg)
        client_msg = request(cid=self.cid, server_msg=server_msg)
        return serde.parameters_res_from_proto(client_msg.parameters_res)

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)
        server_msg = ServerMessage(fit_ins=fit_ins_msg)
        client_msg = request(cid=self.cid, server_msg=server_msg)
        return serde.fit_res_from_proto(client_msg.fit_res)

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        server_msg = ServerMessage(evaluate_ins=evaluate_msg)
        client_msg = request(cid=self.cid, server_msg=server_msg)
        return serde.evaluate_res_from_proto(client_msg.evaluate_res)

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        reconnect_msg = serde.reconnect_to_proto(reconnect)
        server_msg = ServerMessage(reconnect=reconnect_msg)
        client_msg = request(cid=self.cid, server_msg=server_msg)
        return serde.disconnect_from_proto(client_msg.disconnect)


def request(cid: str, server_msg: ServerMessage) -> ClientMessage:
    """Instruct a client to do some work and return the result.

    The `request` function tells the TaskManager that there is a task
    for a particular client (via `set_task`) and then waits until a
    result for this task is available (via `get_result`).
    """

    # Get the current TaskManager
    state = State.instance()
    tm = state.get_task_manager()

    # Set task
    task_set = tm.set_task(cid, server_msg)
    if not task_set:
        return None

    # Wait for result
    client_message = tm.get_result(cid)

    return client_message
