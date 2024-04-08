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
"""Flower ClientProxy implementation using gRPC bidirectional streaming."""


from typing import Optional

from flwr import common
from flwr.common import serde
from flwr.common.aws import BucketManager
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    InsWrapper,
    ResWrapper,
)


class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
        bucket_manager: Optional[BucketManager] = None,
    ):
        super().__init__(cid)
        self.bridge = bridge
        self.bucket_manager = bucket_manager

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetPropertiesRes:
        """Request client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        server_msg = ServerMessage(get_properties_ins=get_properties_msg)
        ins_wrapper = InsWrapper(server_msg, timeout=timeout)
        res_wrapper = self.bridge.request(ins_wrapper)
        client_msg = res_wrapper.raw_message_singular()
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        server_msg = ServerMessage(get_parameters_ins=get_parameters_msg)
        ins_wrapper = InsWrapper(server_msg, timeout=timeout)
        res_wrapper = self.bridge.request(ins_wrapper)
        client_msg_stream = res_wrapper.raw_message_stream()
        get_parameters_res = serde.get_parameters_res_from_proto_stream(
            map(lambda msg: msg.get_parameters_res_stream, client_msg_stream),
            self.bucket_manager,
        )
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        if self.bucket_manager is not None:
            ins.parameters.upload_to_s3(self.bucket_manager)
        fit_ins_msg = serde.fit_ins_to_proto_stream(ins)
        server_msg = map(lambda msg: ServerMessage(fit_ins_stream=msg), fit_ins_msg)
        ins_wrapper = InsWrapper(server_msg, timeout=timeout)
        res_wrapper = self.bridge.request(ins_wrapper)
        client_msg_stream = res_wrapper.raw_message_stream()
        fit_res = serde.fit_res_from_proto_stream(
            map(lambda msg: msg.fit_res_stream, client_msg_stream), self.bucket_manager
        )
        return fit_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        if self.bucket_manager is not None:
            ins.parameters.upload_to_s3(self.bucket_manager)
        evaluate_msg = serde.evaluate_ins_to_proto_stream(ins)
        server_msg = map(
            lambda msg: ServerMessage(evaluate_ins_stream=msg),
            evaluate_msg,
        )
        ins_wrapper = InsWrapper(server_msg, timeout)
        res_wrapper = self.bridge.request(ins_wrapper)
        client_msg: ClientMessage = res_wrapper.raw_message_singular()
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        server_msg = ServerMessage(reconnect_ins=reconnect_ins_msg)
        ins_wrapper = InsWrapper(server_msg, timeout)
        res_wrapper = self.bridge.request(ins_wrapper)
        client_msg = res_wrapper.raw_message_singular()
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect
