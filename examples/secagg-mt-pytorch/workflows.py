import random
from logging import WARNING
from typing import Callable, Dict, Generator, List

import numpy as np

from flwr.common import (
    Parameters,
    Scalar,
    bytes_to_ndarray,
    log,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    generate_shared_key,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_extract,
    get_parameters_shape,
    get_zero_parameters,
    parameters_addition,
    parameters_divide,
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import dequantize
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
)
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
from flwr.common.serde import named_values_from_proto, named_values_to_proto
from flwr.common.typing import Value
from flwr.proto.task_pb2 import SecureAggregation, Task

from task import IS_VALIDATION


def get_workflow_factory() -> (
    Callable[[Parameters, List[int]], Generator[Dict[int, Task], Dict[int, Task], None]]
):
    return _wrap_workflow_with_sec_agg


def _wrap_in_task(named_values: Dict[str, Value]) -> Task:
    return Task(sa=SecureAggregation(named_values=named_values_to_proto(named_values)))


def _get_from_task(task: Task) -> Dict[str, Value]:
    return named_values_from_proto(task.sa.named_values)


_secure_aggregation_configuration = {
    KEY_SHARE_NUMBER: 3,
    KEY_THRESHOLD: 2,
    KEY_CLIPPING_RANGE: 3.0,
    KEY_TARGET_RANGE: 1 << 16,
    KEY_MOD_RANGE: 1 << 30,
}


def workflow_with_sec_agg(
    parameters: Parameters,
    sampled_node_ids: List[int],
    sec_agg_config: Dict[str, Scalar],
) -> Generator[Dict[int, Task], Dict[int, Task], None]:
    """
    =============== Setup stage ===============
    """

    # Protocol config
    num_samples = len(sampled_node_ids)
    num_shares = sec_agg_config[KEY_SHARE_NUMBER]
    threshold = sec_agg_config[KEY_THRESHOLD]
    mod_range = sec_agg_config[KEY_MOD_RANGE]
    # Quantization config
    clipping_range = sec_agg_config[KEY_CLIPPING_RANGE]
    target_range = sec_agg_config[KEY_TARGET_RANGE]

    cfg = {
        KEY_STAGE: STAGE_SETUP,
        KEY_SAMPLE_NUMBER: num_samples,
        KEY_SHARE_NUMBER: num_shares,
        KEY_THRESHOLD: threshold,
        KEY_CLIPPING_RANGE: clipping_range,
        KEY_TARGET_RANGE: target_range,
        KEY_MOD_RANGE: mod_range,
    }
    # The number of shares should better be odd in the SecAgg+ protocol.
    if num_samples != num_shares and num_shares & 0x1 == 0:
        log(WARNING, "Number of shares in the SecAgg+ protocol should be odd.")
        num_shares += 1

    # Randomly assign secure IDs to clients
    sids = [i for i in range(len(sampled_node_ids))]
    random.shuffle(sids)
    nid2sid = dict(zip(sampled_node_ids, sids))
    sid2nid = {sid: nid for nid, sid in nid2sid.items()}
    # Build neighbour relations (node ID -> secure IDs of neighbours)
    half_share = num_shares >> 1
    nid2neighbours = {
        node_id: {
            (nid2sid[node_id] + offset) % num_samples
            for offset in range(-half_share, half_share + 1)
        }
        for node_id in sampled_node_ids
    }

    surviving_node_ids = sampled_node_ids
    # Send setup configuration to clients
    yield {
        node_id: _wrap_in_task(
            named_values={
                **cfg,
                KEY_SECURE_ID: nid2sid[node_id],
            }
        )
        for node_id in surviving_node_ids
    }
    # Receive public keys from clients and build the dict
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]

    sid2public_keys = {}
    for node_id, task in node_messages.items():
        key_dict = _get_from_task(task)
        pk1, pk2 = key_dict[KEY_PUBLIC_KEY_1], key_dict[KEY_PUBLIC_KEY_2]
        sid2public_keys[nid2sid[node_id]] = [pk1, pk2]

    """
    =============== Share keys stage ===============   
    """
    # Braodcast public keys to clients
    yield {
        node_id: _wrap_in_task(
            named_values={
                KEY_STAGE: STAGE_SHARE_KEYS,
                **{
                    str(sid): value
                    for sid, value in sid2public_keys.items()
                    if sid in nid2neighbours[node_id]
                },
            }
        )
        for node_id in surviving_node_ids
    }

    # Receive secret key shares from clients
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]
    # Build forward packet list dictionary
    srcs, dsts, ciphertexts = [], [], []
    fwd_ciphertexts: Dict[int, List[bytes]] = {
        nid2sid[nid]: [] for nid in surviving_node_ids
    }  # dest secure ID -> list of ciphertexts
    fwd_srcs: Dict[int, List[bytes]] = {
        sid: [] for sid in fwd_ciphertexts
    }  # dest secure ID -> list of src secure IDs
    for node_id, task in node_messages.items():
        res_dict = _get_from_task(task)
        srcs += [nid2sid[node_id]] * len(res_dict[KEY_DESTINATION_LIST])
        dsts += res_dict[KEY_DESTINATION_LIST]
        ciphertexts += res_dict[KEY_CIPHERTEXT_LIST]

    for src, dst, ciphertext in zip(srcs, dsts, ciphertexts):
        if dst in fwd_ciphertexts:
            fwd_ciphertexts[dst].append(ciphertext)
            fwd_srcs[dst].append(src)

    """
    =============== Collect masked input stage ===============   
    """

    # Send encrypted secret key shares to clients (plus model parameters)
    weights = parameters_to_ndarrays(parameters)
    yield {
        node_id: _wrap_in_task(
            named_values={
                KEY_STAGE: STAGE_COLLECT_MASKED_INPUT,
                KEY_CIPHERTEXT_LIST: fwd_ciphertexts[nid2sid[node_id]],
                KEY_SOURCE_LIST: fwd_srcs[nid2sid[node_id]],
                KEY_PARAMETERS: [ndarray_to_bytes(arr) for arr in weights],
            }
        )
        for node_id in surviving_node_ids
    }
    # Collect masked input from clients
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]
    # Get shape of vector sent by first client
    masked_vector = [np.array([0], dtype=int)] + get_zero_parameters(
        [w.shape for w in weights]
    )
    # Add all collected masked vectors and compuute available and dropout clients set
    dead_sids = {
        nid2sid[node_id]
        for node_id in sampled_node_ids
        if node_id not in surviving_node_ids
    }
    active_sids = {nid2sid[node_id] for node_id in surviving_node_ids}
    for _, task in node_messages.items():
        named_values = _get_from_task(task)
        client_masked_vec = named_values[KEY_MASKED_PARAMETERS]
        client_masked_vec = [bytes_to_ndarray(b) for b in client_masked_vec]
        masked_vector = parameters_addition(masked_vector, client_masked_vec)

    masked_vector = parameters_mod(masked_vector, mod_range)

    """
    =============== Unmask stage ===============   
    """

    # Send secure IDs of active and dead clients.
    yield {
        node_id: _wrap_in_task(
            named_values={
                KEY_STAGE: STAGE_UNMASK,
                KEY_DEAD_SECURE_ID_LIST: list(dead_sids & nid2neighbours[node_id]),
                KEY_ACTIVE_SECURE_ID_LIST: list(active_sids & nid2neighbours[node_id]),
            }
        )
        for node_id in surviving_node_ids
    }
    # Collect key shares from clients
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]
    # Build collected shares dict
    collected_shares_dict: Dict[int, List[bytes]] = {}
    for nid in sampled_node_ids:
        collected_shares_dict[nid2sid[nid]] = []

    if len(surviving_node_ids) < threshold:
        raise Exception("Not enough available clients after unmask vectors stage")
    for _, task in node_messages.items():
        named_values = _get_from_task(task)
        for owner_sid, share in zip(
            named_values[KEY_SECURE_ID_LIST], named_values[KEY_SHARE_LIST]
        ):
            collected_shares_dict[owner_sid].append(share)
    # Remove mask for every client who is available before ask vectors stage,
    # Divide vector by first element
    active_sids, dead_sids = set(active_sids), set(dead_sids)
    for sid, share_list in collected_shares_dict.items():
        if len(share_list) < threshold:
            raise Exception(
                "Not enough shares to recover secret in unmask vectors stage"
            )
        secret = combine_shares(share_list)
        if sid in active_sids:
            # The seed for PRG is the private mask seed of an active client.
            private_mask = pseudo_rand_gen(
                secret, mod_range, get_parameters_shape(masked_vector)
            )
            masked_vector = parameters_subtraction(masked_vector, private_mask)
        else:
            # The seed for PRG is the secret key 1 of a dropped client.
            neighbor_list = list(nid2neighbours[sid2nid[sid]])
            neighbor_list.remove(sid)

            for neighbor_sid in neighbor_list:
                shared_key = generate_shared_key(
                    bytes_to_private_key(secret),
                    bytes_to_public_key(sid2public_keys[neighbor_sid][0]),
                )
                pairwise_mask = pseudo_rand_gen(
                    shared_key, mod_range, get_parameters_shape(masked_vector)
                )
                if sid > neighbor_sid:
                    masked_vector = parameters_addition(masked_vector, pairwise_mask)
                else:
                    masked_vector = parameters_subtraction(masked_vector, pairwise_mask)
    recon_parameters = parameters_mod(masked_vector, mod_range)
    # Divide vector by number of clients who have given us their masked vector
    # i.e. those participating in final unmask vectors stage
    total_weights_factor, recon_parameters = factor_extract(recon_parameters)
    recon_parameters = parameters_divide(recon_parameters, total_weights_factor)
    aggregated_vector = dequantize(
        quantized_parameters=recon_parameters,
        clipping_range=clipping_range,
        target_range=target_range,
    )
    if IS_VALIDATION:
        print(aggregated_vector[:10])
    aggregated_parameters = ndarrays_to_parameters(aggregated_vector)
    # Update model parameters
    parameters.tensors = aggregated_parameters.tensors
    parameters.tensor_type = aggregated_parameters.tensor_type


def _wrap_workflow_with_sec_agg(
    parameters: Parameters, sampled_node_ids: List[int]
) -> Generator[Dict[int, Task], Dict[int, Task], None]:
    return workflow_with_sec_agg(
        parameters, sampled_node_ids, sec_agg_config=_secure_aggregation_configuration
    )
