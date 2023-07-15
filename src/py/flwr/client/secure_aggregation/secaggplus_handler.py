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
"""Message handler for the SecAgg+ protocol."""


import os
from logging import INFO, WARNING
from typing import List

import numpy as np

from flwr.common import ndarray_to_bytes
from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.shamir import create_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    decrypt,
    encrypt,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.secure_aggregation.secaggplus import (
    pseudo_rand_gen,
    share_keys_plaintext_concat,
    share_keys_plaintext_separate,
)
from flwr.common.secure_aggregation.weights_arithmetic import (
    factor_weights_combine,
    weights_addition,
    weights_mod,
    weights_multiply,
    weights_subtraction,
)
from flwr.common.typing import SecureAggregation

from .handler import SecureAggregationHandler

_stages = ["setup", "share keys", "collect masked input", "unmasking"]


class SecAggPlusHandler(SecureAggregationHandler):
    """Message handler for the SecAgg+ protocol."""

    def handle_secure_aggregation(self, sa: SecureAggregation):
        """Handle incoming message and return results, following the SecAgg+ protocol.

        Parameters
        ----------
        sa : SecureAggregation
            The SecureAggregation sub-message in Task message received
            from the server containing a dictionary of named values.

        Returns
        -------
        SecureAggregation
            The final/intermediate results of the SecAgg+ protocol.
        """
        stage = sa.named_values.pop("stage")
        if not hasattr(self, "current_stage"):
            self.current_stage = "unmasking"
        if stage == "setup":
            if self.current_stage != "unmasking":
                log(WARNING, "restart from setup stage")
            self.current_stage = stage
            return _setup(self, sa)
        # if stage is not "setup", the new stage should be the next stage
        expected_new_stage = _stages[_stages.index(self.current_stage) + 1]
        if stage == expected_new_stage:
            self.current_stage = stage
        else:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_new_stage} stage, but receive {stage} stage"
            )

        if stage == "share keys":
            return _share_keys(self, sa)
        if stage == "collect masked input":
            return _collect_masked_input(self, sa)
        if stage == "unmasking":
            return _unmasking(self, sa)
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(self, sa: SecureAggregation) -> SecureAggregation:
    # Assigning parameter values to object fields
    sec_agg_param_dict = sa.named_values
    self.sample_num = sec_agg_param_dict["share_num"]
    self.sid = sec_agg_param_dict["secure_id"]
    # self.sec_agg_id = sec_agg_param_dict["secure_id"]
    log(INFO, f"Client {self.sid}: starting stage 0...")

    self.share_num = sec_agg_param_dict["share_num"]
    self.threshold = sec_agg_param_dict["threshold"]
    self.drop_flag = sec_agg_param_dict["test_drop"]
    self.clipping_range = sec_agg_param_dict["clipping_range"]
    self.target_range = sec_agg_param_dict["target_range"]
    self.mod_range = sec_agg_param_dict["mod_range"]

    # key is the secure id of another client (int)
    # value is the share of that client's secret (bytes)
    self.b_share_dict = {}
    self.sk1_share_dict = {}
    self.shared_key_2_dict = {}
    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    self.sk1, self.pk1 = generate_key_pairs()
    self.sk2, self.pk2 = generate_key_pairs()

    self.sk1, self.pk1 = private_key_to_bytes(self.sk1), public_key_to_bytes(self.pk1)
    self.sk2, self.pk2 = private_key_to_bytes(self.sk2), public_key_to_bytes(self.pk2)
    log(INFO, f"Client {self.sid}: stage 0 completes. uploading public keys...")
    return SecureAggregation(named_values={"pk1": self.pk1, "pk2": self.pk2})


def _share_keys(self, sa: SecureAggregation) -> SecureAggregation:
    key_dict = sa.named_values
    key_dict = {int(sid): (pk1, pk2) for sid, (pk1, pk2) in key_dict.items()}
    log(INFO, f"Client {self.sid}: starting stage 1...")
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    self.public_keys_dict = key_dict
    # check size is larger than threshold
    if len(self.public_keys_dict) < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # check if all public keys received are unique
    pk_list: List[bytes] = []
    for i in self.public_keys_dict.values():
        pk_list.append(i[0])
        pk_list.append(i[1])
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # sanity check that own public keys are correct in dict
    if (
        self.public_keys_dict[self.sid][0] != self.pk1
        or self.public_keys_dict[self.sid][1] != self.pk2
    ):
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate private mask seed
    self.b = os.urandom(32)

    # Create shares
    b_shares = create_shares(self.b, self.threshold, self.share_num)
    sk1_shares = create_shares(self.sk1, self.threshold, self.share_num)

    srcs, dsts, ciphertexts = [], [], []

    for idx, p in enumerate(self.public_keys_dict.items()):
        client_sid, client_public_keys = p
        if client_sid == self.sid:
            self.b_share_dict[self.sid] = b_shares[idx]
            self.sk1_share_dict[self.sid] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                bytes_to_private_key(self.sk2),
                bytes_to_public_key(client_public_keys[1]),
            )
            self.shared_key_2_dict[client_sid] = shared_key
            plaintext = share_keys_plaintext_concat(
                self.sid, client_sid, b_shares[idx], sk1_shares[idx]
            )
            ciphertext = encrypt(shared_key, plaintext)
            srcs.append(self.sid)
            dsts.append(client_sid)
            ciphertexts.append(ciphertext)

    log(INFO, f"Client {self.sid}: stage 1 completes. uploading key shares...")
    return SecureAggregation(named_values={"dsts": dsts, "ciphertexts": ciphertexts})


def _collect_masked_input(self, sa: SecureAggregation) -> SecureAggregation:
    log(INFO, f"Client {self.sid}: starting stage 2...")
    # Receive shares and fit model
    available_clients: List[int] = []
    ciphertexts = sa.named_values["ciphertexts"]
    srcs = sa.named_values["srcs"]
    assert isinstance(ciphertexts, list)
    if len(ciphertexts) + 1 < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # decode all packets and verify all packets are valid. Save shares received
    for src, ct in zip(srcs, ciphertexts):
        shared_key = self.shared_key_2_dict[src]
        plaintext = decrypt(shared_key, ct)
        _src, dst, b_share, sk1_share = share_keys_plaintext_separate(plaintext)
        available_clients.append(src)
        if src != _src:
            raise ValueError(
                f"Client {self.sid}: received ciphertext from {_src} instead of {src}"
            )
        if dst != self.sid:
            ValueError(
                f"Client {self.sid}: received an encrypted message"
                f"for Client {dst} from Client {src}"
            )
        self.b_share_dict[src] = b_share
        self.sk1_share_dict[src] = sk1_share

    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    if self.drop_flag:
        # log(ERROR, "Force dropout due to testing!!")
        raise Exception("Force dropout due to testing")

    # weights = [bytes_to_ndarray(w) for w in msg.named_values["parameters"]]
    # weights, weights_factor, _ = self.fit(weights, {})

    weights = [np.zeros(10000)]
    weights_factor = 1

    # Quantize weight update vector
    quantized_weights = quantize(weights, self.clipping_range, self.target_range)

    quantized_weights = weights_multiply(quantized_weights, weights_factor)
    quantized_weights = factor_weights_combine(weights_factor, quantized_weights)

    dimensions_list: List[tuple] = [a.shape for a in quantized_weights]

    # add private mask
    private_mask = pseudo_rand_gen(self.b, self.mod_range, dimensions_list)
    quantized_weights = weights_addition(quantized_weights, private_mask)

    for client_id in available_clients:
        # add pairwise mask
        shared_key = generate_shared_key(
            bytes_to_private_key(self.sk1),
            bytes_to_public_key(self.public_keys_dict[client_id][0]),
        )
        # print('shared key length: %d' % len(shared_key))
        pairwise_mask = pseudo_rand_gen(shared_key, self.mod_range, dimensions_list)
        if self.sid > client_id:
            quantized_weights = weights_addition(quantized_weights, pairwise_mask)
        else:
            quantized_weights = weights_subtraction(quantized_weights, pairwise_mask)

    # Take mod of final weight update vector and return to server
    quantized_weights = weights_mod(quantized_weights, self.mod_range)
    # return ndarrays_to_parameters(quantized_weights)
    log(INFO, f"Client {self.sid}: stage 2 completes. uploading masked weights...")
    return SecureAggregation(
        named_values={
            "masked_weights": [ndarray_to_bytes(arr) for arr in quantized_weights]
        }
    )


def _unmasking(self, sa: SecureAggregation) -> SecureAggregation:
    active_sids, dead_sids = (
        sa.named_values["active_sids"],
        sa.named_values["dead_sids"],
    )
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(active_sids) < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    sids, shares = [], []
    sids += active_sids
    shares += [self.b_share_dict[sid] for sid in active_sids]
    sids += dead_sids
    shares += [self.sk1_share_dict[sid] for sid in dead_sids]

    return SecureAggregation(named_values={"sids": sids, "shares": shares})
