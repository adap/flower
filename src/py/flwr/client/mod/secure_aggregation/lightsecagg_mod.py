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


from logging import ERROR, INFO, WARNING
from typing import Dict, List

import galois
import numpy as np

from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_public_key,
    generate_key_pairs,
    generate_shared_key,
    public_key_to_bytes,
)
from flwr.common.secure_aggregation.lightsecagg_utils import (
    compute_aggregated_encoded_mask,
    encode_mask,
    mask_weights,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    get_zero_parameters,
    parameters_multiply,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.typing import Scalar


# set up configurations and return the public key
def setup_config(client: SAClientWrapper, config_dict: Dict[str, Scalar]) -> bytes:
    # Assigning parameter values to object fields
    cfg = config_dict
    client.sec_id = cfg["id"]
    client.N = cfg["sample_num"]
    client.id = cfg["id"]
    client.T = cfg["privacy_guarantee"]
    client.U = cfg["min_clients"]
    client.p = cfg["prime_number"]
    client.clipping_range = cfg["clipping_range"]
    client.target_range = cfg["target_range"]
    client.max_weights_factor = cfg["max_weights_factor"]

    # Testing , to be removed================================================
    client.test = 0
    if "test" in cfg and cfg["test"] == 1:
        client.test = 1
        client.test_vector_shape = [(cfg["test_vector_dimension"],)]
        client.test_dropout_value = cfg["test_dropout_value"]
        client.vector_length = client.test_vector_shape[0][0]
        client.d = padding(client.vector_length + 1, client.U, client.T)
    else:
        # End =================================================================
        weights = parameters_to_ndarrays(client.get_parameters({}))
        client.vector_length = sum([o.size for o in weights])
        client.d = padding(client.vector_length + 1, client.U, client.T)
    # dict key is the ID of another client (int)
    # dict value is the shared key, generated from the secret key and a public key of another client
    client.shared_key_dict = {}
    # dict key is the ID of another client (int)
    # dict value is the encoded sub-mask z_j,i where j is the key, i is the id of current client.
    client.encoded_mask_dict = {}
    client.sk, client.pk = generate_key_pairs()
    log(INFO, "LightSecAgg Stage 0 Completed: Config Set Up")
    return public_key_to_bytes(client.pk)


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
