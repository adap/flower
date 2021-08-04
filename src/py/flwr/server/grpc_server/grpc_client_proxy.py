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


from typing import List, Tuple
from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.common.typing import AskVectorsIns, AskVectorsRes, SetupParamIns, ShareKeysIns, ShareKeysRes, UnmaskVectorsIns, UnmaskVectorsRes
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

    def setup_param(self, setup_param_ins: SetupParamIns):
        setup_param_msg = serde.setup_param_ins_to_proto(setup_param_ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(sec_agg_msg=setup_param_msg)
        )
        serde.check_error(client_msg.sec_agg_res)
        setup_param_res = serde.setup_param_res_from_proto(client_msg.sec_agg_res)
        return setup_param_res

    def ask_keys(self) -> common.AskKeysRes:
        ask_keys_msg = serde.ask_keys_to_proto()
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(sec_agg_msg=ask_keys_msg)
        )
        serde.check_error(client_msg.sec_agg_res)
        ask_keys_res = serde.ask_keys_res_from_proto(client_msg.sec_agg_res)
        return ask_keys_res

    def share_keys(self, share_keys_ins: ShareKeysIns) -> ShareKeysRes:
        share_keys_msg = serde.share_keys_ins_to_proto(share_keys_ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(sec_agg_msg=share_keys_msg)
        )
        serde.check_error(client_msg.sec_agg_res)
        share_keys_res = serde.share_keys_res_from_proto(client_msg.sec_agg_res)
        return share_keys_res

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        ask_vectors_msg = serde.ask_vectors_ins_to_proto(ask_vectors_ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(sec_agg_msg=ask_vectors_msg)
        )
        serde.check_error(client_msg.sec_agg_res)
        ask_vectors_res = serde.ask_vectors_res_from_proto(client_msg.sec_agg_res)
        return ask_vectors_res

    def unmask_vectors(self, unmask_vectors_ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
        unmask_vectors_msg = serde.unmask_vectors_ins_to_proto(unmask_vectors_ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(sec_agg_msg=unmask_vectors_msg)
        )
        serde.check_error(client_msg.sec_agg_res)
        unmask_vectors_res = serde.unmask_vectors_res_from_proto(client_msg.sec_agg_res)
        return unmask_vectors_res

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
