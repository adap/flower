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
from flwr.common import serde, SecretsManager
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_bridge import GRPCBridge
from flwr.common.logger import log
from logging import WARNING


from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.handshakepatterns.interactive.XK import XKHandshakePattern
from dissononce.cipher.aesgcm import AESGCMCipher
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.hash.sha256 import SHA256Hash
from dissononce.exceptions.decrypt import DecryptFailedException


class GrpcClientProxy(ClientProxy):
    """Flower client proxy which delegates over the network using gRPC."""

    def __init__(
        self,
        cid: str,
        bridge: GRPCBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge
        self.auth = 0 

    def authenticate(self):
        """Authenticate the client using XX Handshake"""
        # prepare handshake objects
        server_handshakestate = HandshakeState(
            SymmetricState(
                CipherState(
                    AESGCMCipher()
                ),
                SHA256Hash()
            ),
            X25519DH()
        )
        # initialize handshakestate objects
        server_handshakestate.initialize(XKHandshakePattern(), False, b'', s=SecretsManager.server_key_pair())

        # signal handshake
        params: common.Parameters = common.Parameters([], "handshake0")
        fit_ins_msg = serde.fit_ins_to_proto(common.FitIns(params, {}))
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(fit_ins=fit_ins_msg)
        )
        fit_res_msg = serde.fit_res_from_proto(client_msg.fit_res)

        try: 
            # <- e, es
            server_handshakestate.read_message(fit_res_msg.parameters.tensors[0], bytearray())

            # -> e, ee
            message_buffer = bytearray()
            server_handshakestate.write_message(b'', message_buffer)
            params: common.Parameters = common.Parameters([bytes(message_buffer)], "handshake1")
            fit_ins_msg = serde.fit_ins_to_proto(common.FitIns(params, {}))
            client_msg: ClientMessage = self.bridge.request(
                ServerMessage(fit_ins=fit_ins_msg)
            )
            fit_res_msg = serde.fit_res_from_proto(client_msg.fit_res)

            # <- s, se
            server_handshakestate.read_message(fit_res_msg.parameters.tensors[0], bytearray())
            print("handshake complete")
            self.auth = 1
        except DecryptFailedException:
            log(WARNING, f"Unknown client trying to connect with public key {server_handshakestate.rs.data.hex()}")
            self.auth = -1
            

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        if not self.auth: 
            self.authenticate()

        get_parameters_msg = serde.get_parameters_to_proto()
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(get_parameters=get_parameters_msg)
        )
        parameters_res = serde.parameters_res_from_proto(client_msg.parameters_res)
        return parameters_res

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""
        if not self.auth: 
            self.authenticate()
        fit_ins_msg = serde.fit_ins_to_proto(ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(fit_ins=fit_ins_msg)
        )
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        if not self.auth: 
            self.authenticate()
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(evaluate_ins=evaluate_msg)
        )
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        if not self.auth: 
            self.authenticate()
        reconnect_msg = serde.reconnect_to_proto(reconnect)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(reconnect=reconnect_msg)
        )
        disconnect = serde.disconnect_from_proto(client_msg.disconnect)
        return disconnect
