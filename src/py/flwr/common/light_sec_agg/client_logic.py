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
import galois
from mpc_functions import mask_encoding, compute_aggregate_encoded_mask, model_masking
import numpy as np
from flwr.common.parameter import parameters_to_weights, weights_to_parameters
from flwr.common.sec_agg.sec_agg_primitives import check_clipping_range
from flwr.common.typing import LightSecAggSetupConfigIns, LightSecAggSetupConfigRes, AskEncryptedEncodedMasksIns, \
    AskEncryptedEncodedMasksRes, EncryptedEncodedMasksPacket, Parameters, AskMaskedModelsIns, AskMaskedModelsRes, \
    AskAggregatedEncodedMasksIns, AskAggregatedEncodedMasksRes
from flwr.common.sec_agg import sec_agg_primitives
from flwr_crypto_cpp import create_shares
from flwr.common.logger import log
from logging import DEBUG, ERROR, INFO, WARNING
from typing import Dict, List, Tuple
from flwr.common import (
    AskKeysRes,
)
import timeit


def padding(d, U, T):
    ret = d % (U - T)
    if ret != 0:
        ret = U - T - ret
    return d + ret


def encrypt_sub_mask(key, sub_mask):
    ret = weights_to_parameters([sub_mask])
    plaintext = ret.tensors[0]
    return sec_agg_primitives.encrypt(key, plaintext)


def decrypt_sub_mask(key, ciphertext):
    plaintext = sec_agg_primitives.decrypt(key, ciphertext)
    ret = parameters_to_weights(Parameters(
        tensors=[plaintext],
        tensor_type="numpy.ndarray"
    ))
    return ret[0]


def setup_config(client, ins: LightSecAggSetupConfigIns):
    total_time = -timeit.default_timer()
    # Assigning parameter values to object fields
    cfg = ins.sec_agg_cfg_dict
    client.N = cfg['sample_num']
    client.id = cfg['id']
    client.T = cfg['privacy_guarantee']
    client.U = cfg['min_clients']
    client.p = cfg['prime_number']
    client.clipping_range = cfg['clipping_range']
    client.target_range = cfg['target_range']
    client.max_weights_factor = cfg['max_weights_factor']

    # Testing , to be removed================================================
    client.test = 0
    if 'test' in cfg and cfg['test'] == 1:
        client.test = 1
        client.test_vector_shape = [(cfg['test_vector_dimension'],)]
        client.test_dropout_value = cfg['test_dropout_value']
        client.vector_length = client.test_vector_shape[0][0]
        client.d = padding(client.vector_length, client.U, client.T)
    # End =================================================================
    else:
        weights = parameters_to_weights(client.get_parameters())
        client.vector_length = sum([o.size for o in weights])
        client.d = padding(client.vector_length, client.U, client.T)
    # dict key is the sec_agg_id of another client (int)
    # dict value is the shared key, generated from the secret key and a public key of another client
    client.shared_key_dict = {}
    # dict key is the sec_agg_id of another client (int)
    # dict value is the encoded sub-mask z_j,i where j is the key, i is the id of current client.
    client.encoded_mask_dict = {}
    client.sk, client.pk = sec_agg_primitives.generate_key_pairs()
    log(INFO, "LightSecAgg Stage 0 Completed: Config Set Up")
    total_time = total_time+timeit.default_timer()
    if client.sec_agg_id == 3:
        f = open("log.txt", "a")
        f.write(f"Client without communication stage 0:{total_time} \n")
        f.close()
    return LightSecAggSetupConfigRes(
        pk=sec_agg_primitives.public_key_to_bytes(client.pk)
    )


def ask_encrypted_encoded_masks(client, ins: AskEncryptedEncodedMasksIns):
    total_time = -timeit.default_timer()
    # build key dict
    for other_id, res in ins.public_keys_dict.items():
        if other_id == client.id:
            continue
        pk = sec_agg_primitives.bytes_to_public_key(res.pk)
        client.shared_key_dict[other_id] = sec_agg_primitives.generate_shared_key(client.sk, pk)

    # gen masks
    client.GF = galois.GF(client.p)
    client.msk = np.random.randint(client.p, size=(client.d, 1))
    encoded_msk_set = mask_encoding(client.d, client.N, client.U, client.T, client.GF, client.msk)

    # create packets
    packets = []
    for i in ins.public_keys_dict.keys():
        if i == client.id:
            client.encoded_mask_dict[i] = encoded_msk_set[i]
            continue
        packet = EncryptedEncodedMasksPacket(
            source=client.id,
            destination=i,
            ciphertext=encrypt_sub_mask(client.shared_key_dict[i], encoded_msk_set[i])
        )
        packets.append(packet)
    log(INFO, "SecAgg Stage 1 Completed: Sent Encrypted Sub-masks via Packets")
    total_time = total_time+timeit.default_timer()
    if client.sec_agg_id == 3:
        f = open("log.txt", "a")
        f.write(f"Client without communication stage 1:{total_time} \n")
        f.close()
    return AskEncryptedEncodedMasksRes(packets)


def ask_masked_models(client, ins: AskMaskedModelsIns):
    total_time = -timeit.default_timer()
    packets = ins.packet_list
    fit_ins = ins.fit_ins
    active_clients: List[int] = []

    if len(packets) + 1 < client.U:
        raise Exception("Available neighbours number smaller than threshold")

    # receive and decrypt sub-masks
    for packet in packets:
        source = packet.source
        active_clients.append(source)
        assert packet.destination == client.id
        sub_mask = decrypt_sub_mask(client.shared_key_dict[source], packet.ciphertext)
        client.encoded_mask_dict[source] = sub_mask
    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    '''
    fit_res = client.fit(fit_ins)
    weights = parameters_to_weights(fit_res.parameters)
    weights_factor = fit_res.num_examples
    '''
    # temporary code=========================================================
    if client.test == 1:
        if client.sec_agg_id % 20 < client.test_dropout_value:
            log(ERROR, "Force dropout due to testing!!")
            raise Exception("Force dropout due to testing")
        weights = sec_agg_primitives.weights_zero_generate(
            client.test_vector_shape)
     # IMPORTANT NEED SOME FUNCTION TO GET CORRECT WEIGHT FACTOR
    # NOW WE HARD CODE IT AS 1
    # Generally, should be fit_res.num_examples

    weights_factor = 1

    # END =================================================================

    quantized_weights = sec_agg_primitives.quantize(weights, client.clipping_range, client.target_range)

    # weights factor should not exceed maximum
    if weights_factor > client.max_weights_factor:
        weights_factor = client.max_weights_factor
        log(WARNING, "weights_factor exceeds allowed range and has been clipped. Either increase max_weights_factor, or train with fewer data. (Or server is performing unweighted aggregation)")

    quantized_weights = sec_agg_primitives.weights_multiply(
        quantized_weights, weights_factor)
    quantized_weights = sec_agg_primitives.factor_weights_combine(
        weights_factor, quantized_weights)

    quantized_weights = model_masking(quantized_weights, client.msk, client.GF)
    log(INFO, "LightSecAgg Stage 2 Completed: Sent Masked Models")
    total_time = total_time+timeit.default_timer()
    if client.sec_agg_id == 3:
        f = open("log.txt", "a")
        f.write(f"Client without communication stage 2:{total_time} \n")
        f.close()
    return AskMaskedModelsRes(weights_to_parameters(quantized_weights))


def ask_aggregated_encoded_masks(client, ins: AskAggregatedEncodedMasksIns):
    total_time = -timeit.default_timer()

    active_clients = ins.surviving_clients
    agg_msk = compute_aggregate_encoded_mask(client.encoded_mask_dict, client.GF, active_clients)
    log(INFO, "SecAgg Stage 3 Completed: Sent Aggregated Encoded Masks for Unmasking")

    total_time = total_time+timeit.default_timer()
    if client.sec_agg_id == 3:
        f = open("log.txt", "a")
        f.write(f"Client without communication stage 3:{total_time} \n")
        f.close()
    return AskAggregatedEncodedMasksRes(weights_to_parameters([agg_msk]))

