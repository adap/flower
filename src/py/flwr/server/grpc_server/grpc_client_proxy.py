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


from typing import Optional

from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_bridge import GrpcBridge, InsWrapper, ResWrapper
from typing import List, Tuple

class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
    ) -> common.GetPropertiesRes:
        """Requests client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_properties_ins=get_properties_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parameters_ins=get_parameters_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parameters_res = serde.get_parameters_res_from_proto(
            client_msg.get_parameters_res
        )
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float],
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(evaluate_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(reconnect_ins=reconnect_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect
    
    # Custom Request
    def request(self, question: str, l: List[int], timeout: Optional[float]) -> Tuple[str, int]:
        request_msg = serde.example_msg_to_proto(question, l) # serialize 
        print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(example_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        print(f"client msg: {client_msg}")
        response, answer = serde.example_res_from_proto(client_msg.example_res) # deserialize
        return response, answer
    


    # Step 1) Server sends shared vector_a to clients and they all send back vector_b
    def request_vec_b(self, vector_a: List[int], timeout: Optional[float]) -> List[int]:
        request_msg = serde.shared_vec_a_to_proto(vector_a) # serialize 
        #print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_vector_a_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        #print(f"client msg: {client_msg}")
        vector_b = serde.pub_key_b_from_proto(client_msg.send_vector_b_res) # deserialize
        return vector_b
    


    # Step 2) Server sends aggregated publickey allpub to clients and receive boolean confirmation
    def request_allpub_confirmation(self, aggregated_pubkey: List[int], timeout: Optional[float]) -> bool:
        request_msg = serde.aggregated_pubkey_to_proto(aggregated_pubkey) # serialize 
        #print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_allpub_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        #print(f"client msg: {client_msg}")
        allpub_confirmed = serde.pubkey_confirmation_from_proto(client_msg.send_allpub_res) # deserialize
        return allpub_confirmed
    


    # Step 3) After round, encrypt flat list of parameters into two lists (c0, c1)
    def request_encrypted_parameters(self, request: str, timeout: Optional[float]) -> Tuple[List[int], List[int]]:
        request_msg = serde.request_encrypted_to_proto(request) # serialize 
        #print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(request_encrypted_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        #print(f"client msg: {client_msg}")
        c0, c1 = serde.send_encrypted_from_proto(client_msg.send_encrypted_res) # deserialize
        return c0, c1
    

    # Step 4) Send c1sum to clients and send back decryption share
    def request_decryption_share(self, csum1: List[int], timeout: Optional[float]) -> List[int]:
        request_msg = serde.send_csum1_to_proto(csum1) # serialize 
        #print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_csum_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        #print(f"client msg: {client_msg}")
        d = serde.send_decryption_share_from_proto(client_msg.send_dec_share_res) # deserialize
        return d
    

    # Step 5) Send updated model weights to clients and return confirmation
    def request_modelupdate_confirmation(self, new_weights: List[int], timeout: Optional[float]) -> bool:
        request_msg = serde.send_new_weights_to_proto(new_weights) # serialize 
        #print(f"Received message from server: {request_msg}")
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_new_weights_ins=request_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        #print(f"client msg: {client_msg}")
        confirm = serde.send_update_confirmation_from_proto(client_msg.send_new_weights_res) # deserialize
        return confirm