import random
from logging import WARNING
from typing import Callable, Dict, Generator, List, Optional

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
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import dequantize, quantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    Key,
    Stage,
    RECORD_KEY_CONFIGS,
)
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
from flwr.common.typing import ConfigsRecordValues, FitIns
from flwr.proto.task_pb2 import Task
from flwr.common import serde
from flwr.common.constant import MESSAGE_TYPE_FIT
from flwr.common import RecordSet
from flwr.common import recordset_compat as compat
from flwr.common import ConfigsRecord


LOG_EXPLAIN = True


def get_workflow_factory() -> (
    Callable[[Parameters, List[int]], Generator[Dict[int, Task], Dict[int, Task], None]]
):
    return _wrap_workflow_with_sec_agg


def _wrap_in_task(
    named_values: Dict[str, ConfigsRecordValues], fit_ins: Optional[FitIns] = None
) -> Task:
    if fit_ins is not None:
        recordset = compat.fitins_to_recordset(fit_ins, keep_input=True)
    else:
        recordset = RecordSet()
    recordset.configs_records[RECORD_KEY_CONFIGS] = ConfigsRecord(named_values)
    return Task(
        task_type=MESSAGE_TYPE_FIT,
        recordset=serde.recordset_to_proto(recordset),
    )


def _get_from_task(task: Task) -> Dict[str, ConfigsRecordValues]:
    recordset = serde.recordset_from_proto(task.recordset)
    return recordset.configs_records[RECORD_KEY_CONFIGS]


_secure_aggregation_configuration = {
    Key.SHARE_NUMBER: 3,
    Key.THRESHOLD: 2,
    Key.CLIPPING_RANGE: 3.0,
    Key.TARGET_RANGE: 1 << 20,
    Key.MOD_RANGE: 1 << 30,
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
    num_shares = sec_agg_config[Key.SHARE_NUMBER]
    threshold = sec_agg_config[Key.THRESHOLD]
    mod_range = sec_agg_config[Key.MOD_RANGE]
    # Quantization config
    clipping_range = sec_agg_config[Key.CLIPPING_RANGE]
    target_range = sec_agg_config[Key.TARGET_RANGE]

    if LOG_EXPLAIN:
        _quantized = quantize(
            [np.ones(3) for _ in range(num_samples)], clipping_range, target_range
        )
        print(
            "\n\n################################ Introduction ################################\n"
            "In the example, each client will upload a vector [1.0, 1.0, 1.0] instead of\n"
            "model updates for demonstration purposes.\n"
            "Client 0 is configured to drop out before uploading the masked vector.\n"
            f"After quantization, the raw vectors will be:"
        )
        for i in range(1, num_samples):
            print(f"\t{_quantized[i]} from Client {i}")
        print(
            f"Numbers are rounded to integers stochastically during the quantization\n"
            ", and thus not all entries are identical."
        )
        print(
            "The above raw vectors are hidden from the driver through adding masks.\n"
        )
        print(
            "########################## Secure Aggregation Start ##########################"
        )
    cfg = {
        Key.STAGE: Stage.SETUP,
        Key.SAMPLE_NUMBER: num_samples,
        Key.SHARE_NUMBER: num_shares,
        Key.THRESHOLD: threshold,
        Key.CLIPPING_RANGE: clipping_range,
        Key.TARGET_RANGE: target_range,
        Key.MOD_RANGE: mod_range,
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
    if LOG_EXPLAIN:
        print(
            f"Sending configurations to {num_samples} clients and allocating secure IDs..."
        )
    # Send setup configuration to clients
    yield {
        node_id: _wrap_in_task(
            named_values={
                **cfg,
                Key.SECURE_ID: nid2sid[node_id],
            }
        )
        for node_id in surviving_node_ids
    }
    # Receive public keys from clients and build the dict
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]

    if LOG_EXPLAIN:
        print(f"Received public keys from {len(surviving_node_ids)} clients.")

    sid2public_keys = {}
    for node_id, task in node_messages.items():
        key_dict = _get_from_task(task)
        pk1, pk2 = key_dict[Key.PUBLIC_KEY_1], key_dict[Key.PUBLIC_KEY_2]
        sid2public_keys[nid2sid[node_id]] = [pk1, pk2]

    """
    =============== Share keys stage ===============   
    """
    if LOG_EXPLAIN:
        print(f"\nForwarding public keys...")
    # Broadcast public keys to clients
    yield {
        node_id: _wrap_in_task(
            named_values={
                Key.STAGE: Stage.SHARE_KEYS,
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
    if LOG_EXPLAIN:
        print(f"Received encrypted key shares from {len(surviving_node_ids)} clients.")
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
        srcs += [nid2sid[node_id]] * len(res_dict[Key.DESTINATION_LIST])
        dsts += res_dict[Key.DESTINATION_LIST]
        ciphertexts += res_dict[Key.CIPHERTEXT_LIST]

    for src, dst, ciphertext in zip(srcs, dsts, ciphertexts):
        if dst in fwd_ciphertexts:
            fwd_ciphertexts[dst].append(ciphertext)
            fwd_srcs[dst].append(src)

    """
    =============== Collect masked input stage ===============   
    """

    if LOG_EXPLAIN:
        print(f"\nForwarding encrypted key shares and requesting masked input...")
    # Send encrypted secret key shares to clients (plus model parameters)
    yield {
        node_id: _wrap_in_task(
            named_values={
                Key.STAGE: Stage.COLLECT_MASKED_INPUT,
                Key.CIPHERTEXT_LIST: fwd_ciphertexts[nid2sid[node_id]],
                Key.SOURCE_LIST: fwd_srcs[nid2sid[node_id]],
            },
            fit_ins=FitIns(
                parameters=parameters, config={"drop": nid2sid[node_id] == 0}
            ),
        )
        for node_id in surviving_node_ids
    }
    # Collect masked input from clients
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]
    # Get shape of vector sent by first client
    weights = parameters_to_ndarrays(parameters)
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
    if LOG_EXPLAIN:
        for sid in dead_sids:
            print(f"Client {sid} dropped out.")
    for node_id, task in node_messages.items():
        named_values = _get_from_task(task)
        client_masked_vec = named_values[Key.MASKED_PARAMETERS]
        client_masked_vec = [bytes_to_ndarray(b) for b in client_masked_vec]
        if LOG_EXPLAIN:
            print(f"Received {client_masked_vec[1]} from Client {nid2sid[node_id]}.")
        masked_vector = parameters_addition(masked_vector, client_masked_vec)
    masked_vector = parameters_mod(masked_vector, mod_range)
    """
    =============== Unmask stage ===============   
    """

    if LOG_EXPLAIN:
        print("\nRequesting key shares to unmask the aggregate vector...")
    # Send secure IDs of active and dead clients.
    yield {
        node_id: _wrap_in_task(
            named_values={
                Key.STAGE: Stage.UNMASK,
                Key.DEAD_SECURE_ID_LIST: list(dead_sids & nid2neighbours[node_id]),
                Key.ACTIVE_SECURE_ID_LIST: list(active_sids & nid2neighbours[node_id]),
            }
        )
        for node_id in surviving_node_ids
    }
    # Collect key shares from clients
    node_messages = yield
    surviving_node_ids = [node_id for node_id in node_messages]
    if LOG_EXPLAIN:
        print(f"Received key shares from {len(surviving_node_ids)} clients.")
    # Build collected shares dict
    collected_shares_dict: Dict[int, List[bytes]] = {}
    for nid in sampled_node_ids:
        collected_shares_dict[nid2sid[nid]] = []

    if len(surviving_node_ids) < threshold:
        raise Exception("Not enough available clients after unmask vectors stage")
    for _, task in node_messages.items():
        named_values = _get_from_task(task)
        for owner_sid, share in zip(
            named_values[Key.SECURE_ID_LIST], named_values[Key.SHARE_LIST]
        ):
            collected_shares_dict[owner_sid].append(share)
    # Remove mask for every client who is available before ask vectors stage,
    # divide vector by first element
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
    if LOG_EXPLAIN:
        print(f"Unmasked sum of vectors (quantized): {recon_parameters[0]}")
    # recon_parameters = parameters_divide(recon_parameters, total_weights_factor)
    aggregated_vector = dequantize(
        quantized_parameters=recon_parameters,
        clipping_range=clipping_range,
        target_range=target_range,
    )
    aggregated_vector[0] -= (len(active_sids) - 1) * clipping_range
    if LOG_EXPLAIN:
        print(f"Unmasked sum of vectors (dequantized): {aggregated_vector[0]}")
        print(
            f"Aggregate vector using FedAvg: {aggregated_vector[0] / len(active_sids)}"
        )
        print(
            "########################### Secure Aggregation End ###########################\n\n"
        )
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
