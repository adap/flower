# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Modifier for the LightSecAgg protocol."""


from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, WARNING
from typing import List, cast

import numpy as np
from galois import GF

import flwr.common.recordset_compat as compat
from flwr.client.typing import ClientAppCallable
from flwr.common import (
    ConfigsRecord,
    Context,
    Message,
    MessageType,
    Parameters,
    RecordSet,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_public_key,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.common.secure_aggregation.lightsecagg_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.common.secure_aggregation.lightsecagg_utils import (
    compute_aggregated_encoded_mask,
    decrypt_sub_mask,
    encode_mask,
    encrypt_sub_mask,
    mask_weights,
    ndarray_from_bytes,
    ndarray_to_bytes,
    padding,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_multiply,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.typing import NDArrayInt


@dataclass
class LightSecAggState:
    """State of the LightSecAgg protocol."""

    current_stage: str = Stage.SETUP
    nid: int = 0

    # Protocol settings
    N: int = 0  # The number of all clients
    T: int = 0  # The maximum number of corrupted clients
    U: int = 0  # The minimum number of surviving clients
    p: int = 0  # The prime number for the finite field
    d: int = 0  # The model size after padding

    # Quantization settings
    clipping_range: float = 0.0
    target_range: int = 0
    max_weight: float = 0.0  # The maximum weight factor allowed

    # Runtime variables
    pk: bytes = b""  # ECDH public key
    sk: bytes = b""  # ECDH secret key
    # The dict key is the node ID of another client (int)
    # The dict value is the shared secret with that client from ECDH agreement
    ss_dict: dict[int, bytes] = field(default_factory=dict)
    # The dict key is the node ID of another client (int)
    # The dict value is the encoded sub-mask from that client
    encoded_sub_mask_dict: dict[int, NDArrayInt] = field(default_factory=dict)
    mask: NDArrayInt = field(default=np.ndarray([], dtype=np.int_))

    def to_recordset(self, rs: RecordSet) -> None:
        """Save the instance to the given RecordSet."""
        # Init the ConfigsRecord
        cfg = ConfigsRecord()
        flds = set(self.__dict__.keys())

        # Save unsupported types
        flds.difference_update({"ss_dict", "encoded_sub_mask_dict", "mask"})
        cfg["ss_dict:K"] = list(self.ss_dict.keys())
        cfg["ss_dict:V"] = list(self.ss_dict.values())
        cfg["encoded_sub_mask_dict:K"] = list(self.encoded_sub_mask_dict.keys())
        cfg["encoded_sub_mask_dict:V"] = [
            ndarray_to_bytes(v) for v in self.encoded_sub_mask_dict.values()
        ]
        cfg["mask"] = ndarray_to_bytes(self.mask)

        # Save supported types
        for fld in flds:
            cfg[fld] = self.__dict__[fld]

        # Save to the RecordSet
        rs.configs_records[RECORD_KEY_STATE] = cfg

    @classmethod
    def from_recordset(cls, rs: RecordSet) -> LightSecAggState:
        """Construct an instance from the given RecordSet."""
        # Retrieve the ConfigsRecord
        state = LightSecAggState()
        if RECORD_KEY_STATE not in rs.configs_records:
            return state
        cfg = rs.configs_records[RECORD_KEY_STATE]

        # Retrieve unsupported types
        keys = cast(list[str], cfg.pop("ss_dict:K"))
        values = cast(list[bytes], cfg.pop("ss_dict:V"))
        state.ss_dict = dict(zip(keys, values))
        keys = cast(list[str], cfg.pop("encoded_sub_mask_dict:K"))
        values = cast(list[bytes], cfg.pop("encoded_sub_mask_dict:V"))
        state.encoded_sub_mask_dict = dict(
            zip(keys, [ndarray_from_bytes(v) for v in values])
        )
        state.mask = ndarray_from_bytes(cfg["mask"])

        # Retrieve supported types
        for k, v in cfg.items():
            state.__dict__[k] = v  # type: ignore
        return state


def lightsecagg_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Handle messages following the LightSecAgg protocol."""
    # Ignore non-fit messages
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctxt)

    # Retrieve local state
    state = LightSecAggState.from_recordset(ctxt.state)

    # Retrieve incoming configs
    configs = msg.content.configs_records[RECORD_KEY_CONFIGS]

    # Check the validity of the next stage
    check_stage(state.current_stage, configs)

    # Update the current stage
    state.current_stage = cast(str, configs.pop(Key.STAGE))

    # Check the validity of the configs based on the current stage
    check_configs(state.current_stage, configs)

    # Execute
    out_content = RecordSet()
    if state.current_stage == Stage.SETUP:
        state.nid = msg.metadata.dst_node_id
        res = _setup(state, configs)
    elif state.current_stage == Stage.EXCHANGE_SUB_MASKS:
        res = _upload_encrypted_encoded_masks(state, configs)
    elif state.current_stage == Stage.COLLECT_MASKED_MODELS:
        out_msg = call_next(msg, ctxt)
        out_content = out_msg.content
        fitres = compat.recordset_to_fitres(out_content, keep_input=True)
        res = _upload_masked_models(
            state, configs, fitres.num_examples, fitres.parameters
        )
        for p_record in out_content.parameters_records.values():
            p_record.clear()
    elif state.current_stage == Stage.UNMASK:
        res = _upload_aggregated_encoded_masks(state, configs)
    else:
        raise ValueError(f"Unknown LightSecAgg stage: {state.current_stage}")

    # Save state
    state.to_recordset(ctxt.state)

    # Return message
    return msg.create_reply(res)


def check_stage(current_stage: str, configs: ConfigsRecord) -> None:
    """Check the validity of the next stage."""
    # Check the existence of Config.STAGE
    if Key.STAGE not in configs:
        raise KeyError(
            f"The required key '{Key.STAGE}' is missing from the ConfigsRecord."
        )

    # Check the value type of the Config.STAGE
    next_stage = configs[Key.STAGE]
    if not isinstance(next_stage, str):
        raise TypeError(
            f"The value for the key '{Key.STAGE}' must be of type {str}, "
            f"but got {type(next_stage)} instead."
        )

    # Check the validity of the next stage
    if next_stage == Stage.SETUP:
        if current_stage != Stage.UNMASK:
            log(WARNING, "Restart from the setup stage")
    # If stage is not "setup",
    # the stage from configs should be the expected next stage
    else:
        stages = Stage.all()
        expected_next_stage = stages[(stages.index(current_stage) + 1) % len(stages)]
        if next_stage != expected_next_stage:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_next_stage} stage, but receive {next_stage} stage"
            )


def check_configs(stage: str, configs: ConfigsRecord) -> None:
    """Check the validity of the configs."""
    ...


# set up configurations and return the public key
def _setup(state: LightSecAggState, configs: ConfigsRecord) -> ConfigsRecord:
    log(DEBUG, "Node %d: starting stage 0...", state.nid)
    # Assigning parameter values to object fields
    state.N = cast(int, configs[Key.SAMPLE_NUMBER])
    state.T = cast(int, configs[Key.PRIVACY_GUARANTEE])
    state.U = cast(int, configs[Key.MIN_ACTIVE_CLIENTS])
    state.p = cast(int, configs[Key.PRIME_NUMBER])
    state.clipping_range = cast(float, configs[Key.CLIPPING_RANGE])
    state.target_range = cast(int, configs[Key.TARGET_RANGE])
    state.max_weight = cast(float, configs[Key.MAX_WEIGHT])

    state.d = cast(int, configs[Key.MODEL_SIZE])
    state.d = padding(state.d + 1, state.U, state.T)

    # Generate ECDH keys
    sk, pk = generate_key_pairs()
    state.sk, state.pk = private_key_to_bytes(sk), public_key_to_bytes(pk)
    log(DEBUG, "Node %d: stage 0 completes. uploading public keys...", state.nid)
    return ConfigsRecord({Key.PUBLIC_KEY: state.pk})


def _upload_encrypted_encoded_masks(
    state: LightSecAggState, configs: ConfigsRecord
) -> ConfigsRecord:
    log(DEBUG, "Node %d: starting stage 1...", state.nid)
    # Generate all shared secrets for symmetric encryption
    public_keys_dict = {int(k): cast(bytes, v) for k, v in configs.items()}
    for nid, pk_bytes in public_keys_dict.items():
        if nid == state.nid:
            continue
        pk = bytes_to_public_key(pk_bytes)
        state.ss_dict[nid] = generate_shared_key(state.sk, pk)

    # Generate the mask and encoded sub-masks
    state.mask = np.random.randint(state.p, size=(state.d, 1))
    encoded_msk_set = encode_mask(
        state.d, state.N, state.U, state.T, GF(state.p), state.mask
    )

    # Encrypt encoded sub-masks
    dsts: list[int] = []
    ciphertexts: list[bytes] = []
    for nid in public_keys_dict.keys():
        if nid == state.nid:
            state.encoded_sub_mask_dict[nid] = encoded_msk_set[nid]
            continue
        dsts.append(nid)
        ciphertexts.append(encrypt_sub_mask(state.ss_dict[nid], encoded_msk_set[nid]))
    log(
        DEBUG,
        "Node %d: stage 1 completes. uploading encrypted encoded sub-masks...",
        state.nid,
    )
    return ConfigsRecord({Key.DESTINATION_LIST: dsts, Key.CIPHERTEXT_LIST: ciphertexts})


def _upload_masked_models(
    state: LightSecAggState,
    configs: ConfigsRecord,
    num_examples: int,
    updated_parameters: Parameters,
) -> ConfigsRecord:
    log(DEBUG, "Node %d: starting stage 2...", state.nid)
    ciphertexts = cast(List[bytes], configs[Key.CIPHERTEXT_LIST])
    srcs = cast(List[int], configs[Key.SOURCE_LIST])

    if len(ciphertexts) + 1 < state.U:
        raise ValueError("Not enough surviving clients.")

    # Decrypt encoded sub-masks
    for src, ciphertext in zip(srcs, ciphertexts):
        sub_mask = decrypt_sub_mask(state.ss_dict[src], ciphertext)
        state.encoded_sub_mask_dict[src] = sub_mask

    # Multiply parameters by weight
    ratio = num_examples / state.max_weight
    if ratio > 1:
        log(
            WARNING,
            "Potential overflow warning: the provided weight (%s) exceeds the specified"
            " max_weight (%s). This may lead to overflow issues.",
            num_examples,
            state.max_weight,
        )
    q_ratio = round(ratio * state.target_range)
    dq_ratio = q_ratio / state.target_range

    parameters = parameters_to_ndarrays(updated_parameters)
    parameters = parameters_multiply(parameters, dq_ratio)

    # Quantize parameter update (vector)
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )

    quantized_parameters = factor_combine(q_ratio, quantized_parameters)

    quantized_parameters = mask_weights(quantized_parameters, state.mask, GF(state.p))
    log(DEBUG, "Node %d: stage 2 completed, uploading masked parameters...", state.nid)
    return ConfigsRecord(
        {Key.MASKED_PARAMETERS: [ndarray_to_bytes(arr) for arr in quantized_parameters]}
    )


def _upload_aggregated_encoded_masks(
    state: LightSecAggState, configs: ConfigsRecord
) -> ConfigsRecord:
    active_nids = cast(List[int], configs[Key.ACTIVE_NODE_ID_LIST])
    log(DEBUG, "Node %d: starting stage 3...", state.nid)
    if len(active_nids) + 1 < state.U:
        raise ValueError("Not enough surviving clients.")

    agg_msk = compute_aggregated_encoded_mask(
        state.encoded_sub_mask_dict, GF(state.p), active_nids
    )
    log(
        DEBUG,
        "Node %d: stage 3 completes. uploading the aggregated encoded mask...",
        state.nid,
    )
    return ConfigsRecord(
        {Key.AGGREGATED_ENCODED_MASK: [ndarray_to_bytes(arr) for arr in agg_msk]}
    )
