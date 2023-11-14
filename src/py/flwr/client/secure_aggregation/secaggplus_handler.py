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
"""Message handler for the SecAgg+ protocol."""


import os
from dataclasses import dataclass, field
from logging import ERROR, INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union, cast

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
from flwr.common.secure_aggregation.secaggplus_constants import (
    KEY_ACTIVE_SECURE_ID_LIST,
    KEY_CIPHERTEXT_LIST,
    KEY_CLIPPING_RANGE,
    KEY_DEAD_SECURE_ID_LIST,
    KEY_DESTINATION_LIST,
    KEY_MASKED_PARAMETERS,
    KEY_MOD_RANGE,
    KEY_PARAMETERS,
    KEY_PUBLIC_KEY_1,
    KEY_PUBLIC_KEY_2,
    KEY_SAMPLE_NUMBER,
    KEY_SECURE_ID,
    KEY_SECURE_ID_LIST,
    KEY_SHARE_LIST,
    KEY_SHARE_NUMBER,
    KEY_SOURCE_LIST,
    KEY_STAGE,
    KEY_TARGET_RANGE,
    KEY_THRESHOLD,
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASK,
    STAGES,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_concat,
    share_keys_plaintext_separate,
)
from flwr.common.typing import FitIns, Value

from .handler import SecureAggregationHandler


@dataclass
# pylint: disable-next=too-many-instance-attributes
class SecAggPlusState:
    """State of the SecAgg+ protocol."""

    sid: int = 0
    sample_num: int = 0
    share_num: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    target_range: int = 0
    mod_range: int = 0

    # Secret key (sk) and public key (pk)
    sk1: bytes = b""
    pk1: bytes = b""
    sk2: bytes = b""
    pk2: bytes = b""

    # Random seed for generating the private mask
    rd_seed: bytes = b""

    rd_seed_share_dict: Dict[int, bytes] = field(default_factory=dict)
    sk1_share_dict: Dict[int, bytes] = field(default_factory=dict)
    # The dict of the shared secrets from sk2
    ss2_dict: Dict[int, bytes] = field(default_factory=dict)
    public_keys_dict: Dict[int, Tuple[bytes, bytes]] = field(default_factory=dict)

    client: Optional[Union[Client, NumPyClient]] = None


class SecAggPlusHandler(SecureAggregationHandler):
    """Message handler for the SecAgg+ protocol."""

    _shared_state = SecAggPlusState()
    _current_stage = STAGE_UNMASK

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
        # Check if self is a client
        if not isinstance(self, (Client, NumPyClient)):
            raise TypeError(
                "The subclass of SecAggPlusHandler must be "
                "the subclass of Client or NumPyClient."
            )

        # Check the validity of the next stage
        check_stage(self._current_stage, named_values)

        # Update the current stage
        self._current_stage = cast(str, named_values.pop(KEY_STAGE))

        # Check the validity of the `named_values` based on the current stage
        check_named_values(self._current_stage, named_values)

        # Execute
        if self._current_stage == STAGE_SETUP:
            self._shared_state = SecAggPlusState(client=self)
            return _setup(self._shared_state, named_values)
        if self._current_stage == STAGE_SHARE_KEYS:
            return _share_keys(self._shared_state, named_values)
        if self._current_stage == STAGE_COLLECT_MASKED_INPUT:
            return _collect_masked_input(self._shared_state, named_values)
        if self._current_stage == STAGE_UNMASK:
            return _unmask(self._shared_state, named_values)
        raise ValueError(f"Unknown secagg stage: {self._current_stage}")


def check_stage(current_stage: str, named_values: Dict[str, Value]) -> None:
    """Check the validity of the next stage."""
    # Check the existence of KEY_STAGE
    if KEY_STAGE not in named_values:
        raise KeyError(
            f"The required key '{KEY_STAGE}' is missing from the input `named_values`."
        )

    # Check the value type of the KEY_STAGE
    next_stage = named_values[KEY_STAGE]
    if not isinstance(next_stage, str):
        raise TypeError(
            f"The value for the key '{KEY_STAGE}' must be of type {str}, "
            f"but got {type(next_stage)} instead."
        )

    # Check the validity of the next stage
    if next_stage == STAGE_SETUP:
        if current_stage != STAGE_UNMASK:
            log(WARNING, "Restart from the setup stage")
    # If stage is not "setup",
    # the stage from `named_values` should be the expected next stage
    else:
        expected_next_stage = STAGES[(STAGES.index(current_stage) + 1) % len(STAGES)]
        if next_stage != expected_next_stage:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_next_stage} stage, but receive {next_stage} stage"
            )


# pylint: disable-next=too-many-branches
def check_named_values(stage: str, named_values: Dict[str, Value]) -> None:
    """Check the validity of the input `named_values`."""
    # Check `named_values` for the setup stage
    if stage == STAGE_SETUP:
        key_type_pairs = [
            (KEY_SAMPLE_NUMBER, int),
            (KEY_SECURE_ID, int),
            (KEY_SHARE_NUMBER, int),
            (KEY_THRESHOLD, int),
            (KEY_CLIPPING_RANGE, float),
            (KEY_TARGET_RANGE, int),
            (KEY_MOD_RANGE, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in named_values:
                raise KeyError(
                    f"Stage {STAGE_SETUP}: the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            # Bool is a subclass of int in Python,
            # so `isinstance(v, int)` will return True even if v is a boolean.
            # pylint: disable-next=unidiomatic-typecheck
            if type(named_values[key]) is not expected_type:
                raise TypeError(
                    f"Stage {STAGE_SETUP}: The value for the key '{key}' "
                    f"must be of type {expected_type}, "
                    f"but got {type(named_values[key])} instead."
                )
    elif stage == STAGE_SHARE_KEYS:
        for key, value in named_values.items():
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], bytes)
                or not isinstance(value[1], bytes)
            ):
                raise TypeError(
                    f"Stage {STAGE_SHARE_KEYS}: "
                    f"the value for the key '{key}' must be a list of two bytes."
                )
    elif stage == STAGE_COLLECT_MASKED_INPUT:
        key_type_pairs = [
            (KEY_CIPHERTEXT_LIST, bytes),
            (KEY_SOURCE_LIST, int),
            (KEY_PARAMETERS, bytes),
        ]
        for key, expected_type in key_type_pairs:
            if key not in named_values:
                raise KeyError(
                    f"Stage {STAGE_COLLECT_MASKED_INPUT}: "
                    f"the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            if not isinstance(named_values[key], list) or any(
                elm
                for elm in cast(List[Any], named_values[key])
                # pylint: disable-next=unidiomatic-typecheck
                if type(elm) is not expected_type
            ):
                raise TypeError(
                    f"Stage {STAGE_COLLECT_MASKED_INPUT}: "
                    f"the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    elif stage == STAGE_UNMASK:
        key_type_pairs = [
            (KEY_ACTIVE_SECURE_ID_LIST, int),
            (KEY_DEAD_SECURE_ID_LIST, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in named_values:
                raise KeyError(
                    f"Stage {STAGE_UNMASK}: "
                    f"the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            if not isinstance(named_values[key], list) or any(
                elm
                for elm in cast(List[Any], named_values[key])
                # pylint: disable-next=unidiomatic-typecheck
                if type(elm) is not expected_type
            ):
                raise TypeError(
                    f"Stage {STAGE_UNMASK}: "
                    f"the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    else:
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(state: SecAggPlusState, named_values: Dict[str, Value]) -> Dict[str, Value]:
    # Assigning parameter values to object fields
    sec_agg_param_dict = named_values
    state.sample_num = cast(int, sec_agg_param_dict[KEY_SAMPLE_NUMBER])
    state.sid = cast(int, sec_agg_param_dict[KEY_SECURE_ID])
    log(INFO, "Client %d: starting stage 0...", state.sid)

    state.share_num = cast(int, sec_agg_param_dict[KEY_SHARE_NUMBER])
    state.threshold = cast(int, sec_agg_param_dict[KEY_THRESHOLD])
    state.clipping_range = cast(float, sec_agg_param_dict[KEY_CLIPPING_RANGE])
    state.target_range = cast(int, sec_agg_param_dict[KEY_TARGET_RANGE])
    state.mod_range = cast(int, sec_agg_param_dict[KEY_MOD_RANGE])

    # Dictionaries containing client secure IDs as keys
    # and their respective secret shares as values.
    state.rd_seed_share_dict = {}
    state.sk1_share_dict = {}
    # Dictionary containing client secure IDs as keys
    # and their respective shared secrets (with this client) as values.
    state.ss2_dict = {}

    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()

    state.sk1, state.pk1 = private_key_to_bytes(sk1), public_key_to_bytes(pk1)
    state.sk2, state.pk2 = private_key_to_bytes(sk2), public_key_to_bytes(pk2)
    log(INFO, "Client %d: stage 0 completes. uploading public keys...", state.sid)
    return {KEY_PUBLIC_KEY_1: state.pk1, KEY_PUBLIC_KEY_2: state.pk2}


# pylint: disable-next=too-many-locals
def _share_keys(
    state: SecAggPlusState, named_values: Dict[str, Value]
) -> Dict[str, Value]:
    named_bytes_tuples = cast(Dict[str, Tuple[bytes, bytes]], named_values)
    key_dict = {int(sid): (pk1, pk2) for sid, (pk1, pk2) in named_bytes_tuples.items()}
    log(INFO, "Client %d: starting stage 1...", state.sid)
    state.public_keys_dict = key_dict

    # Check if the size is larger than threshold
    if len(state.public_keys_dict) < state.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # Check if all public keys are unique
    pk_list: List[bytes] = []
    for pk1, pk2 in state.public_keys_dict.values():
        pk_list.append(pk1)
        pk_list.append(pk2)
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # Check if public keys of this client are correct in the dictionary
    if (
        state.public_keys_dict[state.sid][0] != state.pk1
        or state.public_keys_dict[state.sid][1] != state.pk2
    ):
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate the private mask seed
    state.rd_seed = os.urandom(32)

    # Create shares for the private mask seed and the first private key
    b_shares = create_shares(state.rd_seed, state.threshold, state.share_num)
    sk1_shares = create_shares(state.sk1, state.threshold, state.share_num)

    srcs, dsts, ciphertexts = [], [], []

    # Distribute shares
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
    return {KEY_DESTINATION_LIST: dsts, KEY_CIPHERTEXT_LIST: ciphertexts}


# pylint: disable-next=too-many-locals
def _collect_masked_input(
    state: SecAggPlusState, named_values: Dict[str, Value]
) -> Dict[str, Value]:
    log(INFO, "Client %d: starting stage 2...", state.sid)
    available_clients: List[int] = []
    ciphertexts = cast(List[bytes], named_values[KEY_CIPHERTEXT_LIST])
    srcs = cast(List[int], named_values[KEY_SOURCE_LIST])
    if len(ciphertexts) + 1 < state.threshold:
        raise Exception("Not enough available neighbour clients.")

    # Decrypt ciphertexts, verify their sources, and store shares.
    for src, ciphertext in zip(srcs, ciphertexts):
        shared_key = state.ss2_dict[src]
        plaintext = decrypt(shared_key, ciphertext)
        actual_src, dst, rd_seed_share, sk1_share = share_keys_plaintext_separate(
            plaintext
        )
        available_clients.append(src)
        if src != actual_src:
            raise ValueError(
                f"Client {state.sid}: received ciphertext "
                f"from {actual_src} instead of {src}."
            )
        if dst != state.sid:
            ValueError(
                f"Client {state.sid}: received an encrypted message"
                f"for Client {dst} from Client {src}."
            )
        state.rd_seed_share_dict[src] = rd_seed_share
        state.sk1_share_dict[src] = sk1_share

    # Fit client
    parameters_bytes = cast(List[bytes], named_values[KEY_PARAMETERS])
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
        log(ERROR, "Client %d: fit function is missing.", state.sid)

    # Quantize parameter update (vector)
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )

    quantized_parameters = parameters_multiply(quantized_parameters, parameters_factor)
    quantized_parameters = factor_combine(parameters_factor, quantized_parameters)

    dimensions_list: List[Tuple[int, ...]] = [a.shape for a in quantized_parameters]

    # Add private mask
    private_mask = pseudo_rand_gen(state.rd_seed, state.mod_range, dimensions_list)
    quantized_parameters = parameters_addition(quantized_parameters, private_mask)

    for client_id in available_clients:
        # Add pairwise masks
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
    log(INFO, "Client %d: stage 2 completes. uploading masked parameters...", state.sid)
    return {
        KEY_MASKED_PARAMETERS: [ndarray_to_bytes(arr) for arr in quantized_parameters]
    }


def _unmask(state: SecAggPlusState, named_values: Dict[str, Value]) -> Dict[str, Value]:
    log(INFO, "Client %d: starting stage 3...", state.sid)

    active_sids = cast(List[int], named_values[KEY_ACTIVE_SECURE_ID_LIST])
    dead_sids = cast(List[int], named_values[KEY_DEAD_SECURE_ID_LIST])
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
    return {KEY_SECURE_ID_LIST: sids, KEY_SHARE_LIST: shares}
