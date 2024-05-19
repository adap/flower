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
from logging import ERROR, INFO, WARNING, DEBUG
from typing import Dict, List, cast

import galois
import numpy as np

from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_public_key,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.common.secure_aggregation.lightsecagg_utils import (
    compute_aggregated_encoded_mask,
    encode_mask,
    encrypt_sub_mask,
    decrypt_sub_mask,
    padding,
    ndarray_to_bytes,
    ndarray_from_bytes,
    mask_weights,
)
from flwr.common.secure_aggregation.lightsecagg_constants import Key, Stage, RECORD_KEY_STATE, RECORD_KEY_CONFIGS
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    get_zero_parameters,
    parameters_multiply,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common import RecordSet, ConfigsRecord
from flwr.common.typing import NDArrayInt
from dataclasses import dataclass, field


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
    clipping_range: float = 0.
    target_range: int = 0
    max_weight: float = 0.  # The maximum weight factor allowed
    
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
            cfg[fld] = self.__dict__[fld]  # type: ignore
            
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
        state.encoded_sub_mask_dict = dict(zip(keys, [ndarray_from_bytes(v) for v in values]))
        state.mask = ndarray_from_bytes(cfg["mask"])
        
        # Retrieve supported types
        for k, v in cfg.items():
            state.__dict__[k] = v  # type: ignore
        return state
        


# set up configurations and return the public key
def setup_config(state: LightSecAggState, configs: ConfigsRecord) -> ConfigsRecord:
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


def ask_encrypted_encoded_masks(client, public_keys_dict: Dict[int, bytes]):
    # build key dict
    for other_id, pk_bytes in public_keys_dict.items():
        if other_id == client.id:
            continue
        pk = bytes_to_public_key(pk_bytes)
        client.shared_key_dict[other_id] = generate_shared_key(client.sk, pk)

    # gen masks
    client.GF = galois.GF(client.p)
    client.msk = np.random.randint(client.p, size=(client.d, 1))
    encoded_msk_set = encode_mask(
        client.d, client.N, client.U, client.T, client.GF, client.msk
    )

    # create packets
    packets = []
    for i in public_keys_dict.keys():
        if i == client.id:
            client.encoded_mask_dict[i] = encoded_msk_set[i]
            continue
        packet = (
            client.id,
            i,
            encrypt_sub_mask(client.shared_key_dict[i], encoded_msk_set[i]),
        )
        packets.append(packet)
    return packets


def ask_masked_models(client, packets, fit_ins):
    active_clients: List[int] = []

    if len(packets) + 1 < client.U:
        raise Exception("Available neighbours number smaller than threshold")

    # receive and decrypt sub-masks
    for packet in packets:
        src, dst, ciphertext = packet
        active_clients.append(src)
        assert dst == client.id
        sub_mask = decrypt_sub_mask(client.shared_key_dict[src], ciphertext)
        client.encoded_mask_dict[src] = sub_mask
    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    """
    fit_res = client.fit(fit_ins)
    weights = parameters_to_weights(fit_res.parameters)
    weights_factor = fit_res.num_examples
    """
    # temporary code=========================================================
    if client.test == 1:
        if client.id % 20 < client.test_dropout_value:
            log(ERROR, "Force dropout due to testing!!")
            raise Exception("Force dropout due to testing")
        weights = get_zero_parameters(client.test_vector_shape)
    # IMPORTANT NEED SOME FUNCTION TO GET CORRECT WEIGHT FACTOR
    # NOW WE HARD CODE IT AS 1
    # Generally, should be fit_res.num_examples

    weights_factor = 1

    # END =================================================================
    quantized_weights = quantize(weights, client.clipping_range, client.target_range)

    # weights factor should not exceed maximum
    if weights_factor > client.max_weights_factor:
        weights_factor = client.max_weights_factor
        log(
            WARNING,
            "weights_factor exceeds allowed range and has been clipped. Either increase max_weights_factor, or train with fewer data. (Or server is performing unweighted aggregation)",
        )

    quantized_weights = parameters_multiply(quantized_weights, weights_factor)
    quantized_weights = factor_combine(weights_factor, quantized_weights)

    quantized_weights = mask_weights(quantized_weights, client.msk, client.GF)
    log(INFO, "LightSecAgg Stage 2 Completed: Sent Masked Models")
    return ndarrays_to_parameters(quantized_weights)


def ask_aggregated_encoded_masks(client, active_clients):
    agg_msk = compute_aggregated_encoded_mask(
        client.encoded_mask_dict, client.GF, active_clients
    )
    log(INFO, "SecAgg Stage 3 Completed: Sent Aggregated Encoded Masks for Unmasking")
    return ndarrays_to_parameters([agg_msk])
