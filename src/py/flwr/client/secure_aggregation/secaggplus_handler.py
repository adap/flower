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
from dataclasses import dataclass, field
from logging import ERROR, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union, cast

from flwr.client.client import Client
from flwr.client.numpy_client import NumPyClient
from flwr.common import (
    bytes_to_ndarray,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
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
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_addition,
    parameters_mod,
    parameters_multiply,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_concat,
    share_keys_plaintext_separate,
)
from flwr.common.typing import FitIns, Value

from .handler import SecureAggregationHandler

STAGE_SETUP = "setup"
STAGE_SHARE_KEYS = "share keys"
STAGE_COLLECT_MASKED_INPUT = "collect masked input"
STAGE_UNMASKING = "unmasking"
STAGES = (STAGE_SETUP, STAGE_SHARE_KEYS, STAGE_COLLECT_MASKED_INPUT, STAGE_UNMASKING)


@dataclass
# pylint: disable-next=too-many-instance-attributes
class SecAggPlusState:
    """State of the SecAgg+ protocol."""

    sid: int = 0
    sample_num: int = 0
    share_num: int = 0
    threshold: int = 0
    test_drop: bool = False
    clipping_range: float = 0.0
    target_range: int = 0
    mod_range: int = 0

    # sk, pk stand for secret key, public key
    sk1: bytes = b""
    pk1: bytes = b""
    sk2: bytes = b""
    pk2: bytes = b""
    # random seed for generating the private mask
    rd_seed: bytes = b""

    rd_seed_share_dict: Dict[int, bytes] = field(default_factory=dict)
    sk1_share_dict: Dict[int, bytes] = field(default_factory=dict)
    # the dict of the shared secrets from sk2
    ss2_dict: Dict[int, bytes] = field(default_factory=dict)
    public_keys_dict: Dict[int, Tuple[bytes, bytes]] = field(default_factory=dict)

    client: Optional[Union[Client, NumPyClient]] = None


class SecAggPlusHandler(SecureAggregationHandler):
    """Message handler for the SecAgg+ protocol."""

    _shared_state = SecAggPlusState()
    _current_stage = STAGE_UNMASKING

    def handle_secure_aggregation(
        self, named_values: Dict[str, Value]
    ) -> Dict[str, Value]:
        """Handle incoming message and return results, following the SecAgg+ protocol.

        Parameters
        ----------
        named_values : Dict[str, Value]
            The named values retrieved from the SecureAggregation sub-message
            of Task message in the server's TaskIns.

        Returns
        -------
        Dict[str, Value]
            The final/intermediate results of the SecAgg+ protocol.
        """
        if not isinstance(self, (Client, NumPyClient)):
            raise TypeError(
                "The subclass of SecAggPlusHandler must be "
                "the subclass of Client or NumPyClient."
            )
        stage = str(named_values.pop("stage"))
        if stage == STAGE_SETUP:
            if self._current_stage != STAGE_UNMASKING:
                log(WARNING, "restart from setup stage")
            self._shared_state = SecAggPlusState(client=self)
            self._current_stage = stage
            return _setup(self._shared_state, named_values)
        # if stage is not "setup", the new stage should be the next stage
        expected_new_stage = STAGES[
            (STAGES.index(self._current_stage) + 1) % len(STAGES)
        ]
        if stage == expected_new_stage:
            self._current_stage = stage
        else:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_new_stage} stage, but receive {stage} stage"
            )

        if stage == STAGE_SHARE_KEYS:
            return _share_keys(self._shared_state, named_values)
        if stage == STAGE_COLLECT_MASKED_INPUT:
            return _collect_masked_input(self._shared_state, named_values)
        if stage == STAGE_UNMASKING:
            return _unmasking(self._shared_state, named_values)
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(state: SecAggPlusState, named_values: Dict[str, Value]) -> Dict[str, Value]:
    # Assigning parameter values to object fields
    sec_agg_param_dict = named_values
    state.sample_num = cast(int, sec_agg_param_dict["share_num"])
    state.sid = cast(int, sec_agg_param_dict["secure_id"])
    log(INFO, "Client %d: starting stage 0...", state.sid)

    state.share_num = cast(int, sec_agg_param_dict["share_num"])
    state.threshold = cast(int, sec_agg_param_dict["threshold"])
    state.test_drop = cast(bool, sec_agg_param_dict["test_drop"])
    state.clipping_range = cast(float, sec_agg_param_dict["clipping_range"])
    state.target_range = cast(int, sec_agg_param_dict["target_range"])
    state.mod_range = cast(int, sec_agg_param_dict["mod_range"])

    # key is the secure id of another client (int)
    # value is the share of that client's secret (bytes)
    state.rd_seed_share_dict = {}
    state.sk1_share_dict = {}
    state.ss2_dict = {}
    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()

    state.sk1, state.pk1 = private_key_to_bytes(sk1), public_key_to_bytes(pk1)
    state.sk2, state.pk2 = private_key_to_bytes(sk2), public_key_to_bytes(pk2)
    log(INFO, "Client %d: stage 0 completes. uploading public keys...", state.sid)
    return {"pk1": state.pk1, "pk2": state.pk2}


# pylint: disable-next=too-many-locals
def _share_keys(
    state: SecAggPlusState, named_values: Dict[str, Value]
) -> Dict[str, Value]:
    named_bytes_tuples = cast(Dict[str, Tuple[bytes, bytes]], named_values)
    key_dict = {int(sid): (pk1, pk2) for sid, (pk1, pk2) in named_bytes_tuples.items()}
    log(INFO, "Client %d: starting stage 1...", state.sid)
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    state.public_keys_dict = key_dict
    # check size is larger than threshold
    if len(state.public_keys_dict) < state.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # check if all public keys received are unique
    pk_list: List[bytes] = []
    for pk1, pk2 in state.public_keys_dict.values():
        pk_list.append(pk1)
        pk_list.append(pk2)
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # sanity check that own public keys are correct in dict
    if (
        state.public_keys_dict[state.sid][0] != state.pk1
        or state.public_keys_dict[state.sid][1] != state.pk2
    ):
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate private mask seed
    state.rd_seed = os.urandom(32)

    # Create shares
    b_shares = create_shares(state.rd_seed, state.threshold, state.share_num)
    sk1_shares = create_shares(state.sk1, state.threshold, state.share_num)

    srcs, dsts, ciphertexts = [], [], []

    for idx, (sid, (_, pk2)) in enumerate(state.public_keys_dict.items()):
        if sid == state.sid:
            state.rd_seed_share_dict[state.sid] = b_shares[idx]
            state.sk1_share_dict[state.sid] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                bytes_to_private_key(state.sk2),
                bytes_to_public_key(pk2),
            )
            state.ss2_dict[sid] = shared_key
            plaintext = share_keys_plaintext_concat(
                state.sid, sid, b_shares[idx], sk1_shares[idx]
            )
            ciphertext = encrypt(shared_key, plaintext)
            srcs.append(state.sid)
            dsts.append(sid)
            ciphertexts.append(ciphertext)

    log(INFO, "Client %d: stage 1 completes. uploading key shares...", state.sid)
    return {"dsts": dsts, "ciphertexts": ciphertexts}


# pylint: disable-next=too-many-locals
def _collect_masked_input(
    state: SecAggPlusState, named_values: Dict[str, Value]
) -> Dict[str, Value]:
    log(INFO, "Client %d: starting stage 2...", state.sid)
    # Receive shares and fit model
    available_clients: List[int] = []
    ciphertexts = cast(List[bytes], named_values["ciphertexts"])
    srcs = cast(List[int], named_values["srcs"])
    if len(ciphertexts) + 1 < state.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # decode all packets and verify all packets are valid. Save shares received
    for src, ciphertext in zip(srcs, ciphertexts):
        shared_key = state.ss2_dict[src]
        plaintext = decrypt(shared_key, ciphertext)
        _src, dst, b_share, sk1_share = share_keys_plaintext_separate(plaintext)
        available_clients.append(src)
        if src != _src:
            raise ValueError(
                f"Client {state.sid}: received ciphertext from {_src} instead of {src}"
            )
        if dst != state.sid:
            ValueError(
                f"Client {state.sid}: received an encrypted message"
                f"for Client {dst} from Client {src}"
            )
        state.rd_seed_share_dict[src] = b_share
        state.sk1_share_dict[src] = sk1_share

    # fit client
    parameters_bytes = cast(List[bytes], named_values["parameters"])
    parameters = [bytes_to_ndarray(w) for w in parameters_bytes]
    if isinstance(state.client, Client):
        fit_res = state.client.fit(
            FitIns(parameters=ndarrays_to_parameters(parameters), config={})
        )
        parameters_factor = fit_res.num_examples
        parameters = parameters_to_ndarrays(fit_res.parameters)
    elif isinstance(state.client, NumPyClient):
        parameters, parameters_factor, _ = state.client.fit(parameters, {})
    else:
        log(ERROR, "Client %d: fit function is none", state.sid)

    # Quantize weight update vector
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )

    quantized_parameters = parameters_multiply(quantized_parameters, parameters_factor)
    quantized_parameters = factor_combine(parameters_factor, quantized_parameters)

    dimensions_list: List[Tuple[int, ...]] = [a.shape for a in quantized_parameters]

    # add private mask
    private_mask = pseudo_rand_gen(state.rd_seed, state.mod_range, dimensions_list)
    quantized_parameters = parameters_addition(quantized_parameters, private_mask)

    for client_id in available_clients:
        # add pairwise mask
        shared_key = generate_shared_key(
            bytes_to_private_key(state.sk1),
            bytes_to_public_key(state.public_keys_dict[client_id][0]),
        )
        pairwise_mask = pseudo_rand_gen(shared_key, state.mod_range, dimensions_list)
        if state.sid > client_id:
            quantized_parameters = parameters_addition(
                quantized_parameters, pairwise_mask
            )
        else:
            quantized_parameters = parameters_subtraction(
                quantized_parameters, pairwise_mask
            )

    # Take mod of final weight update vector and return to server
    quantized_parameters = parameters_mod(quantized_parameters, state.mod_range)
    # return ndarrays_to_parameters(quantized_parameters)
    log(INFO, "Client %d: stage 2 completes. uploading masked parameters...", state.sid)
    return {
        "masked_parameters": [ndarray_to_bytes(arr) for arr in quantized_parameters]
    }


def _unmasking(
    state: SecAggPlusState, named_values: Dict[str, Value]
) -> Dict[str, Value]:
    log(INFO, "Client %d: starting stage 3...", state.sid)

    active_sids = cast(List[int], named_values["active_sids"])
    dead_sids = cast(List[int], named_values["dead_sids"])
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(active_sids) < state.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    sids, shares = [], []
    sids += active_sids
    shares += [state.rd_seed_share_dict[sid] for sid in active_sids]
    sids += dead_sids
    shares += [state.sk1_share_dict[sid] for sid in dead_sids]

    log(INFO, "Client %d: stage 3 completes. uploading key shares...", state.sid)
    return {"sids": sids, "shares": shares}
