from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import random
from typing import List, Union, Dict, Generator, Callable

from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    ndarray_to_bytes,
    bytes_to_ndarray
)
from flwr.common.typing import Task, SecureAggregation, ServerMessage, FitIns
from flwr.common.secure_aggregation.sec_agg_protocol import *


def get_workflow_factory() -> (
    Callable[[Parameters, List[int]], Generator[Dict[int, Task], Dict[int, Task], None]]
):
    return _wrap_workflow_with_sec_agg


_secure_aggregation_configuration = {
    "threshold": 2,
    "test_dropouts": 1,
    "clipping_range": 3,
    "target_range": 1 << 16,
    "mod_range": 1 << 24,
}


# def workflow_without_sec_agg(parameters: Parameters, sampled_node_ids: List[int]) \
#         -> Generator[Dict[int, Task], Dict[int, Task], None]:
#     # configure fit
#     fit_ins = FitIns(parameters=parameters, config={})
#     task = Task(legacy_server_message=ServerMessage(fit_ins=fit_ins))
#     yield {node_id: task for node_id in sampled_node_ids}

#     # aggregate fit
#     node_messages: Dict[int, Task] = yield
#     print(f'updating parameters with received messages {node_messages}...')
#     # todo
    
    
# def workflow_with_sec_agg(parameters: Parameters, sampled_node_ids: List[int]) \
#         -> Generator[Dict[int, Task], Dict[int, Task], None]:
            
#     yield request_keys_ins(sampled_node_ids)
    
#     node_messages: Dict[int, Task] = yield
#     yield share_keys_ins(node_messages)
    
#     node_messages: Dict[int, Task] = yield
#     yield request_parameters_ins(node_messages)
    
#     node_messages: Dict[int, Task] = yield
#     yield request_key_shares_ins(sampled_node_ids, node_messages)
    
#     node_messages: Dict[int, Task] = yield
#     print(f'trying to decrypt and update parameters...')
#     # todo


def secagg_workflow_test(
    parameters: Parameters, sampled_node_ids: List[int]
) -> Generator[Dict[int, Task], Dict[int, Task], None]:
    print(
        "========================== START SECAGG_WORKFLOW TEST =========================="
    )
    for i in range(4):
        yield {
            node_id: Task(
                message_type="sec_agg",
                secure_aggregation_message=SecureAggregation(
                    named_scalars={"secagg_round": i}
                ),
            )
            for node_id in sampled_node_ids
        }
        node_responses = yield
        for node_id, task in node_responses.items():
            print(
                f"{node_id}: {task.secure_aggregation_message.named_scalars['secagg_msg']}"
            )
    print(
        "========================== END SECAGG_WORKFLOW TEST =========================="
    )


def workflow_with_sec_agg(
    parameters: Parameters,
    sampled_node_ids: List[int],
    sec_agg_config: Dict[str, Scalar],
) -> Generator[Dict[int, Task], Dict[int, Task], None]:
    stages = ["setup", "share keys", "collect masked input", "unmasking"]

    """
    =============== Setup stage ===============   
    """

    cfg = {
        "stage": stages[0],
        "share_num": len(sampled_node_ids),
        "threshold": sec_agg_config["threshold"],
        "clipping_range": sec_agg_config["clipping_range"],
        "target_range": sec_agg_config["target_range"],
        "mod_range": sec_agg_config["mod_range"],
    }
    # randomly assign secure id to clients
    sids = [i for i in range(len(sampled_node_ids))]
    random.shuffle(sids)
    nid2sid = dict(zip(sampled_node_ids, sids))
    sid2nid = {sid: nid for nid, sid in nid2sid.items()}
    surviving_node_ids = sampled_node_ids
    # send setup configuration to clients
    yield {
        node_id: Task(
            message_type="sec_agg",
            secure_aggregation_message=SecureAggregation(
                named_values={
                    **cfg,
                    "secure_id": nid2sid[node_id],
                    "test_drop": nid2sid[node_id] < sec_agg_config["test_dropouts"],
                }
            ),
        )
        for node_id in surviving_node_ids
    }
    # receive public keys from clients and build the dict
    node_messages = yield
    node_messages = {
        node_id: task
        for node_id, task in node_messages.items()
        if task.message_type != "error"
    }
    surviving_node_ids = [node_id for node_id in node_messages]

    sid2public_keys = {}
    for node_id, task in node_messages.items():
        key_dict = task.secure_aggregation_message.named_values
        pk1, pk2 = key_dict["pk1"], key_dict["pk2"]
        sid2public_keys[nid2sid[node_id]] = [pk1, pk2]

    """
    =============== Share keys stage ===============   
    """

    # braodcast public keys to clients
    braodcast_task = Task(
        message_type="sec_agg",
        secure_aggregation_message=SecureAggregation(
            named_values={
                "stage": stages[1],
                **{str(sid): value for sid, value in sid2public_keys.items()},
            }
        ),
    )
    yield {node_id: braodcast_task for node_id in surviving_node_ids}

    # receive secret key shares from clients
    node_messages = yield
    node_messages = {
        node_id: task
        for node_id, task in node_messages.items()
        if task.message_type != "error"
    }
    surviving_node_ids = [node_id for node_id in node_messages]
    # Build forward packet list dictionary
    srcs, dsts, ciphertexts = [], [], []
    fwd_ciphertexts: Dict[int, List[bytes]] = {
        nid2sid[nid]: [] for nid in node_messages
    }  # dest secure id -> list of ciphertexts
    fwd_srcs: Dict[int, List[bytes]] = {
        sid: [] for sid in fwd_ciphertexts
    }  # dest secure id -> list of src sids
    for node_id, task in node_messages.items():
        res_dict = task.secure_aggregation_message.named_values
        srcs += [nid2sid[node_id]] * len(res_dict["dsts"])
        dsts += res_dict["dsts"]
        ciphertexts += res_dict["ciphertexts"]

    for src, dst, ciphertext in zip(srcs, dsts, ciphertexts):
        if dst in fwd_ciphertexts:
            fwd_ciphertexts[dst].append(ciphertext)
            fwd_srcs[dst].append(src)
    tmp_dict = {sid: len(lst) for sid, lst in fwd_ciphertexts.items()}
    print(f"foward_packet_list_dict: {tmp_dict}")

    """
    =============== Collect masked input stage ===============   
    """

    # send encrypted secret key shares to clients (plus model parameters)
    # weights = parameters_to_ndarrays(parameters)
    weights = [np.zeros(10000)]
    yield {
        node_id: Task(
            message_type="sec_agg",
            secure_aggregation_message=SecureAggregation(
                named_values={
                    "ciphertexts": fwd_ciphertexts[nid2sid[node_id]],
                    "srcs": fwd_srcs[nid2sid[node_id]],
                    "stage": stages[2],
                    "parameters": [ndarray_to_bytes(arr) for arr in weights]
                }
            ),
        )
        for node_id in surviving_node_ids
    }
    # collect masked input from clients
    node_messages = yield
    node_messages = {
        node_id: task
        for node_id, task in node_messages.items()
        if task.message_type != "error"
    }
    surviving_node_ids = [node_id for node_id in node_messages]
    # Get shape of vector sent by first client
    masked_vector = [np.array([0], dtype=int)] + weights_zero_generate(
        [w.shape for w in weights]
    )
    # Add all collected masked vectors and compuute available and dropout clients set
    dead_sids = [
        nid2sid[node_id]
        for node_id in sampled_node_ids
        if node_id not in surviving_node_ids
    ]
    active_sids = [nid2sid[node_id] for node_id in surviving_node_ids]
    for _, task in node_messages.items():
        client_masked_vec = task.secure_aggregation_message.named_values[
            "masked_weights"
        ]
        client_masked_vec = [bytes_to_ndarray(b) for b in client_masked_vec]
        masked_vector = weights_addition(masked_vector, client_masked_vec)

    masked_vector = weights_mod(masked_vector, 1 << 24)

    """
    =============== Unmasking stage ===============   
    """

    # broadcast secure ids of active and dead clients.
    braodcast_task = Task(
        message_type="sec_agg",
        secure_aggregation_message=SecureAggregation(
            named_values={
                "stage": stages[3],
                "dead_sids": dead_sids,
                "active_sids": active_sids,
            }
        ),
    )
    yield {node_id: braodcast_task for node_id in surviving_node_ids}
    # collect key shares from clients
    node_messages = yield
    node_messages = {
        node_id: task
        for node_id, task in node_messages.items()
        if task.message_type != "error"
    }
    surviving_node_ids = [node_id for node_id in node_messages]
    # Build collected shares dict
    collected_shares_dict: Dict[int, List[bytes]] = {}
    for nid in sampled_node_ids:
        collected_shares_dict[nid2sid[nid]] = []

    if len(surviving_node_ids) < sec_agg_config["threshold"]:
        raise Exception("Not enough available clients after unmask vectors stage")
    for _, task in node_messages.items():
        named_values = task.secure_aggregation_message.named_values
        for owner_sid, share in zip(named_values["sids"], named_values["shares"]):
            collected_shares_dict[owner_sid].append(share)
    # Remove mask for every client who is available before ask vectors stage,
    # Divide vector by first element
    active_sids, dead_sids = set(active_sids), set(dead_sids)
    for sid, share_list in collected_shares_dict.items():
        if len(share_list) < sec_agg_config["threshold"]:
            raise Exception(
                "Not enough shares to recover secret in unmask vectors stage"
            )
        secret = combine_shares(share_list)
        if sid in active_sids:
            # seed is an available client's b
            private_mask = pseudo_rand_gen(
                secret, 1 << 24, weights_shape(masked_vector)
            )
            masked_vector = weights_subtraction(masked_vector, private_mask)
        else:
            # seed is a dropout client's sk1
            neighbor_list = list(sid2nid.keys())
            neighbor_list.remove(sid)

            for neighbor_sid in neighbor_list:
                shared_key = generate_shared_key(
                    bytes_to_private_key(secret),
                    bytes_to_public_key(sid2public_keys[neighbor_sid][0]),
                )
                pairwise_mask = pseudo_rand_gen(
                    shared_key, 1 << 24, weights_shape(masked_vector)
                )
                if sid > neighbor_sid:
                    masked_vector = weights_addition(masked_vector, pairwise_mask)
                else:
                    masked_vector = weights_subtraction(masked_vector, pairwise_mask)
    masked_vector = weights_mod(masked_vector, 1 << 24)
    # Divide vector by number of clients who have given us their masked vector
    # i.e. those participating in final unmask vectors stage
    total_weights_factor, masked_vector = factor_weights_extract(masked_vector)
    masked_vector = weights_divide(masked_vector, total_weights_factor)
    aggregated_vector = reverse_quantize(masked_vector, 3, 1 << 16)
    print(aggregated_vector[:10])
    aggregated_parameters = ndarrays_to_parameters(aggregated_vector)
    # update model parameters
    parameters.tensors = aggregated_parameters.tensors
    parameters.tensor_type = aggregated_parameters.tensor_type


def _wrap_workflow_with_sec_agg(
    parameters: Parameters, sampled_node_ids: List[int]
) -> Generator[Dict[int, Task], Dict[int, Task], None]:
    return workflow_with_sec_agg(
        parameters, sampled_node_ids, sec_agg_config=_secure_aggregation_configuration
    )
