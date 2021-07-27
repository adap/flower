from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    secagg_utils,
)
from flwr.common.typing import SetupParamIn, ShareKeysIn, ShareKeysPacket, ShareKeysRes
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

    def setup_param(self, setup_param_in: SetupParamIn):
        self.sample_num = setup_param_in.sample_num
        self.secagg_id = setup_param_in.secagg_id
        self.share_num = setup_param_in.share_num
        self.threshold = setup_param_in.threshold

        # key is the secagg_id of another client
        # value is the secret share we possess that contributes to the client's secret
        self.b_share = {}
        self.sk1_share = {}
        log(INFO, f"SecAgg Params: {setup_param_in}")

    def ask_keys(self):
        self.sk1, self.pk1 = secagg_utils.generate_key_pairs()
        self.sk2, self.pk2 = secagg_utils.generate_key_pairs()
        log(INFO, "Created SecAgg Key Pairs")
        return AskKeysRes(
            pk1=secagg_utils.public_key_to_bytes(self.pk1),
            pk2=secagg_utils.public_key_to_bytes(self.pk2),
        )

    def share_keys(self, share_keys_in: ShareKeysIn) -> ShareKeysRes:
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
                self.b_share[self.secagg_id] = b_shares[idx]
                self.sk1_share[self.secagg_id] = sk1_shares[idx]
            else:
                shared_key = secagg_utils.generate_shared_key(
                    self.sk2, secagg_utils.bytes_to_public_key(client_public_keys.pk2))
                plaintext = secagg_utils.share_keys_plaintext_concat(
                    self.secagg_id, client_secagg_id, b_shares[idx], sk1_shares[idx])
                ciphertext = secagg_utils.encrypt(shared_key, plaintext)
                share_keys_packet = ShareKeysPacket(
                    source=self.secagg_id, destination=client_secagg_id, ciphertext=ciphertext)
                share_keys_res.share_keys_res_list.append(share_keys_packet)

        log(INFO, "Sent shares")
        return share_keys_res
