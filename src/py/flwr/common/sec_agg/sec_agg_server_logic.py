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
from logging import INFO, WARNING
import math
from typing import Dict, List, Optional, Tuple
from flwr.common.parameter import parameters_to_weights, weights_to_parameters
from flwr.common.typing import AskKeysIns, AskKeysRes, AskVectorsIns, AskVectorsRes, FitIns, FitRes, Parameters, Scalar, \
    SetupParamIns, SetupParamRes, ShareKeysIns, ShareKeysPacket, ShareKeysRes, UnmaskVectorsIns, UnmaskVectorsRes
from flwr.server.client_proxy import ClientProxy
from flwr.common.sec_agg import sec_agg_primitives
from flwr.common.secure_aggregation import SAClientMessageCarrier, SAServerMessageCarrier, SecureAggregationFitRound, \
    SecureAggregationResultsAndFailures
from flwr_crypto_cpp import combine_shares
import timeit
import numpy as np
from flwr.common.timer import Timer

from flwr.common.logger import log

SetupParamResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, SetupParamRes]], List[BaseException]
]
AskKeysResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskKeysRes]], List[BaseException]
]
ShareKeysResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, ShareKeysRes]], List[BaseException]
]
AskVectorsResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskVectorsRes]], List[BaseException]
]
UnmaskVectorsResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, UnmaskVectorsRes]], List[BaseException]
]
FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]


# No type annotation for server because of circular dependency!
def sec_agg_fit_round(strategy: SecureAggregationFitRound, server, rnd: int
                      ) -> Optional[
    Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
]:
    tm = Timer()
    # Sample clients
    client_instruction_list = strategy.configure_fit(
        rnd=rnd, parameters=server.parameters, client_manager=server.client_manager())
    proxy2id: Dict[ClientProxy, int] = {}
    setup_param_clients: Dict[int, ClientProxy] = {}
    client_instructions: Dict[int, FitIns] = {}
    for idx, value in enumerate(client_instruction_list):
        proxy2id[value[0]] = idx
        setup_param_clients[idx] = value[0]
        client_instructions[idx] = value[1]

    # Get sec_agg parameters from strategy
    tm.tic()
    log(INFO, "Get sec_agg_param_dict from strategy")
    tm.tic('s0')
    sec_agg_param_dict = server.strategy.config
    sec_agg_param_dict["sample_num"] = len(client_instruction_list)
    sec_agg_param_dict = process_sec_agg_param_dict(sec_agg_param_dict)
    graph: Dict[int, List[int]] = {}
    sample_num = sec_agg_param_dict["sample_num"]
    share_num = sec_agg_param_dict['share_num']
    assert share_num == sample_num or share_num & 1 == 1
    if share_num != sample_num:
        t = share_num >> 1
        for idx in setup_param_clients.keys():
            lst = [(idx + i) % sample_num for i in range(-t, t + 1)]
            graph[idx] = lst
    else:
        lst = [i for i in range(sample_num)]
        for idx in setup_param_clients.keys():
            graph[idx] = lst

    # === Stage 0: Setup ===
    # Give rnd, sample_num, share_num, threshold, client id
    log(INFO, "SecAgg Stage 0: Setting up Params And Asking Keys")
    tm.toc('s0')
    # do not count setup time
    tm.tic('s0_com')
    ask_keys_results_and_failures = setup_param(
        strategy,
        clients=setup_param_clients,
        sec_agg_param_dict=sec_agg_param_dict
    )
    tm.toc('s0_com')
    tm.tic('s0')
    # time consumption of stage 1 on the server side
    public_keys_dict: Dict[int, AskKeysRes] = {}
    ask_keys_results = ask_keys_results_and_failures[0]
    if len(ask_keys_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after ask keys stage")
    share_keys_clients: Dict[int, ClientProxy] = {}

    # Build public keys dict
    # tmp_map = dict(ask_keys_results)
    for client, result in ask_keys_results:
        idx = proxy2id[client]
        public_keys_dict[idx] = result
        share_keys_clients[idx] = client
    # for idx, client in ask_keys_clients.items():
    #     if client in [result[0] for result in ask_keys_results]:
    #         pos = [result[0] for result in ask_keys_results].index(client)
    #         public_keys_dict[idx] = ask_keys_results[pos][1]
    #         share_keys_clients[idx] = client
    # key = tmp_map.get(client, None)
    # if key is not None:
    #     public_keys_dict[idx] = key
    #     share_keys_clients[idx] = client
    tm.toc('s0')

    # === Stage 1: Share Keys ===
    log(INFO, "SecAgg Stage 1: Sharing Keys")
    tm.tic('s1_com')
    share_keys_results_and_failures = share_keys(
        strategy, graph,
        share_keys_clients, public_keys_dict
    )
    tm.toc('s1_com')
    tm.tic('s1')
    share_keys_results = share_keys_results_and_failures[0]
    if len(share_keys_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after share keys stage")

    # Build forward packet list dictionary
    total_packet_list: List[ShareKeysPacket] = []
    forward_packet_list_dict: Dict[int, List[ShareKeysPacket]] = {}  # destination -> list of packets
    ask_vectors_clients: Dict[int, ClientProxy] = {}
    # tmp_map = dict(share_keys_results)
    for idx, client in share_keys_clients.items():
        if client in [result[0] for result in share_keys_results]:
            pos = [result[0] for result in share_keys_results].index(client)
            ask_vectors_clients[idx] = client
            packet_list = share_keys_results[pos][1].share_keys_res_list
            total_packet_list += packet_list
        # o = tmp_map.get(client, None)
        # if o is not None:
        #     ask_vectors_clients[idx] = client
        #     total_packet_list.extend(o.share_keys_res_list)

    for idx in ask_vectors_clients.keys():
        forward_packet_list_dict[idx] = []
    # forward_packet_list_dict = dict(zip(ask_vectors_clients.keys(), [] * len(ask_vectors_clients.keys())))

    for packet in total_packet_list:
        destination = packet.destination
        if destination in ask_vectors_clients.keys():
            forward_packet_list_dict[destination].append(packet)

    tm.toc('s1')

    # === Stage 2: Ask Vectors ===
    log(INFO, "SecAgg Stage 2: Asking Vectors")
    tm.tic('s2_com')
    ask_vectors_results_and_failures = ask_vectors(
        strategy, ask_vectors_clients, forward_packet_list_dict, client_instructions)
    tm.toc('s2_com')
    tm.tic('s2')
    ask_vectors_results = ask_vectors_results_and_failures[0]
    if len(ask_vectors_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after ask vectors stage")
    # Get shape of vector sent by first client
    masked_vector = sec_agg_primitives.weights_zero_generate(
        [i.shape for i in parameters_to_weights(ask_vectors_results[0][1].parameters)])
    # Add all collected masked vectors and compuute available and dropout clients set
    unmask_vectors_clients: Dict[int, ClientProxy] = {}
    dropout_clients = ask_vectors_clients.copy()
    for idx, client in ask_vectors_clients.items():
        if client in [result[0] for result in ask_vectors_results]:
            pos = [result[0] for result in ask_vectors_results].index(client)
            unmask_vectors_clients[idx] = client
            dropout_clients.pop(idx)
            client_parameters = ask_vectors_results[pos][1].parameters
            masked_vector = sec_agg_primitives.weights_addition(
                masked_vector, parameters_to_weights(client_parameters))
    masked_vector = sec_agg_primitives.weights_mod(masked_vector, sec_agg_param_dict['mod_range'])
    tm.toc('s2')
    # === Stage 3: Unmask Vectors ===
    shamir_reconstruction_time = 0
    prg_time = 0
    log(INFO, "SecAgg Stage 3: Unmasking Vectors")
    tm.tic('s3_com')
    unmask_vectors_results_and_failures = unmask_vectors(
        strategy, unmask_vectors_clients, dropout_clients, graph)
    unmask_vectors_results = unmask_vectors_results_and_failures[0]
    tm.toc('s3_com')

    tm.tic('s3')
    # Build collected shares dict
    collected_shares_dict: Dict[int, List[bytes]] = {}
    for idx in ask_vectors_clients.keys():
        collected_shares_dict[idx] = []

    if len(unmask_vectors_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after unmask vectors stage")
    for result in unmask_vectors_results:
        unmask_vectors_res = result[1]
        for owner_id, share in unmask_vectors_res.share_dict.items():
            collected_shares_dict[owner_id].append(share)

    # Remove mask for every client who is available before ask vectors stage,
    # Divide vector by first element
    for client_id, share_list in collected_shares_dict.items():
        if len(share_list) < sec_agg_param_dict['threshold']:
            raise Exception(
                "Not enough shares to recover secret in unmask vectors stage")
        tm.tic('combine_shares')
        secret = combine_shares(share_list)
        tm.toc('combine_shares')
        if client_id in unmask_vectors_clients.keys():
            # seed is an available client's b
            tm.tic('mask_gen')
            private_mask = sec_agg_primitives.pseudo_rand_gen(
                secret, sec_agg_param_dict['mod_range'], sec_agg_primitives.weights_shape(masked_vector))
            tm.toc('mask_gen')
            masked_vector = sec_agg_primitives.weights_subtraction(
                masked_vector, private_mask)
        else:
            # seed is a dropout client's sk1
            neighbor_list: List[int] = []
            if sec_agg_param_dict['share_num'] == sec_agg_param_dict['sample_num']:
                neighbor_list = list(ask_vectors_clients.keys())
                neighbor_list.remove(client_id)
            else:
                # share_num must be odd
                for i in range(-(sec_agg_param_dict['share_num'] >> 1), (sec_agg_param_dict['share_num'] >> 1) + 1):
                    if i != 0 and ((i + client_id) % sec_agg_param_dict['sample_num']) in ask_vectors_clients.keys():
                        neighbor_list.append((i + client_id) %
                                             sec_agg_param_dict['sample_num'])

            for neighbor_id in neighbor_list:
                shared_key = sec_agg_primitives.generate_shared_key(
                    sec_agg_primitives.bytes_to_private_key(secret),
                    sec_agg_primitives.bytes_to_public_key(public_keys_dict[neighbor_id].pk1))
                tm.tic('mask_gen')
                pairwise_mask = sec_agg_primitives.pseudo_rand_gen(
                    shared_key, sec_agg_param_dict['mod_range'], sec_agg_primitives.weights_shape(masked_vector))
                tm.toc('mask_gen')
                if client_id > neighbor_id:
                    masked_vector = sec_agg_primitives.weights_addition(
                        masked_vector, pairwise_mask)
                else:
                    masked_vector = sec_agg_primitives.weights_subtraction(
                        masked_vector, pairwise_mask)
    masked_vector = sec_agg_primitives.weights_mod(
        masked_vector, sec_agg_param_dict['mod_range'])
    # Divide vector by number of clients who have given us their masked vector
    # i.e. those participating in final unmask vectors stage
    total_weights_factor, masked_vector = sec_agg_primitives.factor_weights_extract(
        masked_vector)
    masked_vector = sec_agg_primitives.weights_divide(
        masked_vector, total_weights_factor)
    aggregated_vector = sec_agg_primitives.reverse_quantize(
        masked_vector, sec_agg_param_dict['clipping_range'], sec_agg_param_dict['target_range'])
    print(aggregated_vector[:4])
    aggregated_parameters = weights_to_parameters(aggregated_vector)
    tm.toc('s3')
    tm.toc()
    times = tm.get_all()
    f = open("log.txt", "a")
    f.write(f"Server time with communication:{times['default']} \n")
    f.write(f"Server time without communication:{sum([times['s0'], times['s1'], times['s3']])} \n")
    f.write(f"first element {aggregated_vector[0].flatten()[0]}\n\n\n")
    f.write("server time (detail):\n%s\n" %
            '\n'.join([f"round {i} = {times['s' + str(i)]} ({times['s' + str(i)] * 100. / times['default']:.2f} %)"
                       for i in range(4)]))
    f.write('shamir\'s key reconstruction time: %f (%.2f%%)\n' % (times['combine_shares'],
                                                                  times['combine_shares'] * 100. / times['default']))
    f.write('mask generation time: %f (%.2f%%)\n' % (times['mask_gen'], times['mask_gen'] * 100. / times['default']))
    f.write('num of dropouts: %d\n' % len(dropout_clients))
    f.close()
    return aggregated_parameters, None, None


def process_sec_agg_param_dict(sec_agg_param_dict: Dict[str, Scalar]) -> Dict[str, Scalar]:
    # min_num will be replaced with intended min_num based on sample_num
    # if both min_frac or min_num not provided, we take maximum of either 2 or 0.9 * sampled
    # if either one is provided, we use that
    # Otherwise, we take the maximum
    # Note we will eventually check whether min_num>=2
    if 'min_frac' not in sec_agg_param_dict:
        if 'min_num' not in sec_agg_param_dict:
            sec_agg_param_dict['min_num'] = max(
                2, int(0.9 * sec_agg_param_dict['sample_num']))
    else:
        if 'min_num' not in sec_agg_param_dict:
            sec_agg_param_dict['min_num'] = int(
                sec_agg_param_dict['min_frac'] * sec_agg_param_dict['sample_num'])
        else:
            sec_agg_param_dict['min_num'] = max(sec_agg_param_dict['min_num'], int(
                sec_agg_param_dict['min_frac'] * sec_agg_param_dict['sample_num']))

    if 'share_num' not in sec_agg_param_dict:
        # Complete graph
        sec_agg_param_dict['share_num'] = sec_agg_param_dict['sample_num']
    elif sec_agg_param_dict['share_num'] % 2 == 0 and sec_agg_param_dict['share_num'] != sec_agg_param_dict[
        'sample_num']:
        # we want share_num of each node to be either odd or sample_num
        log(WARNING,
            "share_num value changed due to sample num and share_num constraints! See documentation for reason")
        sec_agg_param_dict['share_num'] += 1

    if 'threshold' not in sec_agg_param_dict:
        sec_agg_param_dict['threshold'] = max(
            2, int(sec_agg_param_dict['share_num'] * 0.9))

    # Maximum number of example trained set to 1000
    if 'max_weights_factor' not in sec_agg_param_dict:
        sec_agg_param_dict['max_weights_factor'] = 1

    # Quantization parameters
    if 'clipping_range' not in sec_agg_param_dict:
        sec_agg_param_dict['clipping_range'] = 3

    if 'target_range' not in sec_agg_param_dict:
        sec_agg_param_dict['target_range'] = 1 << 16

    if 'mod_range' not in sec_agg_param_dict:
        min_bits = math.ceil(math.log2(sec_agg_param_dict['sample_num'] * sec_agg_param_dict['target_range'] *
                                       sec_agg_param_dict['max_weights_factor']))
        sec_agg_param_dict['mod_range'] = 1 << min_bits

    if 'timeout' not in sec_agg_param_dict:
        sec_agg_param_dict['timeout'] = 30

    log(
        INFO,
        f"SecAgg parameters: {sec_agg_param_dict}",
    )

    assert (
        sec_agg_param_dict['sample_num'] >= 2
        and sec_agg_param_dict['min_num'] >= 2
        and sec_agg_param_dict['sample_num'] >= sec_agg_param_dict['min_num']
        and sec_agg_param_dict['share_num'] <= sec_agg_param_dict['sample_num']
        and sec_agg_param_dict['threshold'] <= sec_agg_param_dict['share_num']
        and sec_agg_param_dict['threshold'] >= 2
        and (sec_agg_param_dict['share_num'] % 2 == 1 or sec_agg_param_dict['share_num'] == sec_agg_param_dict[
        'sample_num'])
        and sec_agg_param_dict['target_range'] * sec_agg_param_dict['sample_num'] * sec_agg_param_dict[
            'max_weights_factor'] <= sec_agg_param_dict['mod_range']
    ), "SecAgg parameters not accepted"
    return sec_agg_param_dict


def setup_param(
    strategy: SecureAggregationFitRound,
    clients: Dict[int, ClientProxy],
    sec_agg_param_dict: Dict[str, Scalar]
) -> SetupParamResultsAndFailures:
    def sec_agg_param_dict_with_sec_agg_id(sec_agg_param_dict: Dict[str, Scalar], sec_agg_id: int):
        new_sec_agg_param_dict = sec_agg_param_dict.copy()
        new_sec_agg_param_dict[
            'sec_agg_id'] = sec_agg_id
        return new_sec_agg_param_dict

    results, failures = strategy.sa_request([
        (
            c,
            SAServerMessageCarrier(
                identifier='0',
                str2scalar=sec_agg_param_dict_with_sec_agg_id(
                    sec_agg_param_dict, idx),
            )
        ) for idx, c in clients.items()
    ])
    results = [(c, AskKeysRes(pk1=msg.bytes_list[0], pk2=msg.bytes_list[1])) for c, msg in results]
    return results, failures


# def ask_keys(strategy: SecureAggregationFitRound, clients: Dict[int, ClientProxy]) -> AskKeysResultsAndFailures:
#     results, failures = strategy.sa_request([
#         (c, SAServerMessageCarrier(identifier='1')) for c in clients.values()
#     ])
#     results = [(c, AskKeysRes(pk1=msg.bytes_list[0], pk2=msg.bytes_list[1])) for c, msg in results]
#     return results, failures


def share_keys(strategy: SecureAggregationFitRound, graph: Dict[int, List[int]], clients: Dict[int, ClientProxy],
               public_keys_dict: Dict[int, AskKeysRes]) \
    -> ShareKeysResultsAndFailures:
    results, failures = strategy.sa_request([share_keys_client(c, idx, graph, public_keys_dict)
                                             for idx, c in clients.items()])
    new_results = []
    for client, result in results:
        lst = [ShareKeysPacket(source, destination, ciphertext)
               for source, destination, ciphertext in zip(result.numpy_ndarray_list[0],
                                                          result.numpy_ndarray_list[1],
                                                          result.bytes_list)]
        new_results.append((
            client, ShareKeysRes(lst)
        ))
    return new_results, failures


def share_keys_client(client: ClientProxy, idx: int, graph: Dict[int, List[int]],
                      public_keys_dict: Dict[int, AskKeysRes],
                      ) -> Tuple[ClientProxy, SAServerMessageCarrier]:
    local_dict: Dict[str, AskKeysRes] = {}
    key_set = public_keys_dict.keys()
    for k in graph[idx]:
        if k in key_set:
            v = public_keys_dict[k]
            local_dict[str(k) + '_pk1'] = v.pk1
            local_dict[str(k) + '_pk2'] = v.pk2

    return client, SAServerMessageCarrier(identifier='1', str2scalar=local_dict)


def ask_vectors(strategy: SecureAggregationFitRound, clients: Dict[int, ClientProxy],
                forward_packet_list_dict: Dict[int, List[ShareKeysPacket]],
                client_instructions: Dict[int, FitIns]) -> AskVectorsResultsAndFailures:
    results, failures = strategy.sa_request([ask_vectors_client(c, forward_packet_list_dict[idx],
                                                                client_instructions[idx])
                                             for idx, c in clients.items()])
    results = [(c, AskVectorsRes(r.parameters)) for c, r in results]
    return results, failures


def ask_vectors_client(client: ClientProxy, forward_packet_list: List[ShareKeysPacket], fit_ins: FitIns) \
    -> Tuple[ClientProxy, SAServerMessageCarrier]:
    source_lst = np.array([o.source for o in forward_packet_list])
    destination_lst = np.array([o.destination for o in forward_packet_list])
    ciphertext_lst = [o.ciphertext for o in forward_packet_list]
    msg = SAServerMessageCarrier('2', numpy_ndarray_list=[source_lst, destination_lst],
                                 bytes_list=ciphertext_lst, fit_ins=fit_ins)
    return client, msg


def unmask_vectors(strategy: SecureAggregationFitRound, clients: Dict[int, ClientProxy],
                   dropout_clients: Dict[int, ClientProxy], graph: Dict[int, List[int]]) \
        -> UnmaskVectorsResultsAndFailures:
    actives = set(clients.keys())
    dropouts = set(dropout_clients.keys())

    results, failures = strategy.sa_request([
        (client, SAServerMessageCarrier(
            identifier='3',
            numpy_ndarray_list=[
                np.array(list(actives.intersection(graph[idx]))),
                np.array(list(dropouts.intersection(graph[idx])), dtype=np.int32),
            ]
        ))
        for idx, client in clients.items()
    ])
    results = [(
        c, UnmaskVectorsRes(dict([(int(k), v) for k, v in res.str2scalar.items()]))
    ) for c, res in results]
    return results, failures
