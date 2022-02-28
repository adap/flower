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
"""gRPC-based Flower ClientProxy implementation."""


from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_bridge import GRPCBridge


class GrpcClientProxy(ClientProxy):
    """Flower client proxy which delegates over the network using gRPC."""

    def __init__(
        self,
        cid: str,
        bridge: GRPCBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(self, ins: common.PropertiesIns) -> common.PropertiesRes:
        """Requests client's set of internal properties."""
        properties_msg = serde.properties_ins_to_proto(ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(properties_ins=properties_msg)
        )
        properties_res = serde.properties_res_from_proto(client_msg.properties_res)
        return properties_res

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_to_proto()
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(get_parameters=get_parameters_msg)
        )
        parameters_res = serde.parameters_res_from_proto(client_msg.parameters_res)
        return parameters_res

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(fit_ins=fit_ins_msg)
        )
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(evaluate_ins=evaluate_msg)
        )
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        reconnect_msg = serde.reconnect_to_proto(reconnect)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(reconnect=reconnect_msg)
        )
        disconnect = serde.disconnect_from_proto(client_msg.disconnect)
        return disconnect
