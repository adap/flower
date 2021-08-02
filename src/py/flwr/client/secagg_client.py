import numpy as np
from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    secagg_utils,
)
from flwr.common.parameter import parameters_to_weights, weights_to_parameters
from flwr.common.typing import AskVectorsIns, AskVectorsRes, SetupParamIns, ShareKeysIns, ShareKeysPacket, ShareKeysRes
from flwr.server.strategy import secagg
from .client import Client
from flwr.common.logger import log
from logging import DEBUG, INFO, WARNING


class SecAggClient(Client):
    """Wrapper which adds SecAgg methods."""

    def __init__(self, c: Client) -> None:
        self.client = c

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        return self.client.get_parameters()

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    def setup_param(self, setup_param_in: SetupParamIns):
        self.sample_num = setup_param_in.sample_num
        self.secagg_id = setup_param_in.secagg_id
        self.share_num = setup_param_in.share_num
        self.threshold = setup_param_in.threshold
        self.clipping_range = setup_param_in.clipping_range
        self.target_range = setup_param_in.target_range
        self.mod_range = setup_param_in.mod_range

        # key is the secagg_id of another client
        # value is the secret share we possess that contributes to the client's secret
        self.b_share_dict = {}
        self.sk1_share_dict = {}
        self.shared_key_2_dict = {}
        log(INFO, f"SecAgg Params: {setup_param_in}")

    def ask_keys(self):
        self.sk1, self.pk1 = secagg_utils.generate_key_pairs()
        self.sk2, self.pk2 = secagg_utils.generate_key_pairs()
        log(INFO, "Created SecAgg Key Pairs")
        return AskKeysRes(
            pk1=secagg_utils.public_key_to_bytes(self.pk1),
            pk2=secagg_utils.public_key_to_bytes(self.pk2),
        )

    def share_keys(self, share_keys_in: ShareKeysIns) -> ShareKeysRes:
        self.public_keys_dict = share_keys_in.public_keys_dict
        # check size is larger than threshold
        if len(self.public_keys_dict) < self.threshold:
            raise Exception("Available neighbours number smaller than threshold")

        # check if all public keys received are unique
        pk_list = []
        for i in self.public_keys_dict.values():
            pk_list.append(i.pk1)
            pk_list.append(i.pk2)
        if len(set(pk_list)) != len(pk_list):
            raise Exception("Some public keys are identical")

        # sanity check that own public keys are correct in dict
        if self.public_keys_dict[self.secagg_id].pk1 != secagg_utils.public_key_to_bytes(self.pk1) or self.public_keys_dict[self.secagg_id].pk2 != secagg_utils.public_key_to_bytes(self.pk2):
            raise Exception(
                "Own public keys are displayed in dict incorrectly, should not happen!")

        self.b = secagg_utils.rand_bytes(32)
        b_shares = secagg_utils.create_shares(
            self.b, self.threshold, self.sample_num
        )
        sk1_shares = secagg_utils.create_shares(
            secagg_utils.private_key_to_bytes(self.sk1), self.threshold, self.sample_num
        )

        share_keys_res = ShareKeysRes(share_keys_res_list=[])

        for idx, p in enumerate(self.public_keys_dict.items()):
            client_secagg_id, client_public_keys = p
            if client_secagg_id == self.secagg_id:
                self.b_share_dict[self.secagg_id] = b_shares[idx]
                self.sk1_share_dict[self.secagg_id] = sk1_shares[idx]
            else:
                shared_key = secagg_utils.generate_shared_key(
                    self.sk2, secagg_utils.bytes_to_public_key(client_public_keys.pk2))
                self.shared_key_2_dict[client_secagg_id] = shared_key
                plaintext = secagg_utils.share_keys_plaintext_concat(
                    self.secagg_id, client_secagg_id, b_shares[idx], sk1_shares[idx])
                ciphertext = secagg_utils.encrypt(shared_key, plaintext)
                share_keys_packet = ShareKeysPacket(
                    source=self.secagg_id, destination=client_secagg_id, ciphertext=ciphertext)
                share_keys_res.share_keys_res_list.append(share_keys_packet)

        log(INFO, "Sent shares")
        return share_keys_res

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        packet_list = ask_vectors_ins.ask_vectors_in_list
        fit_ins = ask_vectors_ins.fit_ins
        available_clients = []

        if len(packet_list)+1 < self.threshold:
            raise Exception("Available neighbours number smaller than threshold")

        # decode all packets
        for packet in packet_list:
            source = packet.source
            available_clients.append(source)
            destination = packet.destination
            ciphertext = packet.ciphertext
            if destination != self.secagg_id:
                raise Exception(
                    "Received packet meant for another user. Not supposed to happen")
            shared_key = self.shared_key_2_dict[source]
            plaintext = secagg_utils.decrypt(shared_key, ciphertext)
            try:
                plaintext_source, plaintext_destination, plaintext_b_share, plaintext_sk1_share = secagg_utils.share_keys_plaintext_separate(
                    plaintext)
            except:
                raise Exception(
                    "Decryption of ciphertext failed. Not supposed to happen")
            if plaintext_source != source:
                raise Exception(
                    "Received packet source is different from intended source. Not supposed to happen")
            if plaintext_destination != destination:
                raise Exception(
                    "Received packet destination is different from intended destination. Not supposed to happen")
            self.b_share_dict[source] = plaintext_b_share
            self.sk1_share_dict[source] = plaintext_sk1_share

        # fit client
        # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
        '''
        fit_res = self.client.fit(fit_ins)
        parameters = fit_res.parameters
        weights = parameters_to_weights(parameters)
        '''
        # temporary code
        weights = [np.array([[-0.2, -0.5, 1.9], [0.0, 2.4, -1.9]]),
                   np.array([[0.2, 0.5, -1.9], [0.0, -2.4, 1.9]])]
        quantized_weights = secagg_utils.quantize(
            weights, self.clipping_range, self.target_range)
        for client in available_clients:
            if client == self.secagg_id:
                # add private mask
                pass

            else:
                # add pairwise mask
                pass

        log(INFO, "Sent vectors")
        return AskVectorsRes(parameters=weights_to_parameters(quantized_weights))
