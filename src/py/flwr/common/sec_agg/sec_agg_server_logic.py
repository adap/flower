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
from flwr_crypto_cpp import combine_shares
import timeit
import sys
import concurrent.futures

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


# No type annotation for server because of cirular dependency!
def sec_agg_fit_round(server, rnd: int
                      ) -> Optional[
    Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
]:
    total_time = 0
    total_time = total_time - timeit.default_timer()
    timer = []
    # Sample clients
    client_instruction_list = server.strategy.configure_fit(
        rnd=rnd, parameters=server.parameters, client_manager=server._client_manager)
    setup_param_clients: Dict[int, ClientProxy] = {}
    client_instructions: Dict[int, FitIns] = {}
    for idx, value in enumerate(client_instruction_list):
        setup_param_clients[idx] = value[0]
        client_instructions[idx] = value[1]

    # Get sec_agg parameters from strategy
    log(INFO, "Get sec_agg_param_dict from strategy")
    sec_agg_param_dict = server.strategy.get_sec_agg_param()
    sec_agg_param_dict["sample_num"] = len(client_instruction_list)
    sec_agg_param_dict = process_sec_agg_param_dict(sec_agg_param_dict)

    # === Stage 0: Setup ===
    # Give rnd, sample_num, share_num, threshold, client id
    log(INFO, "SecAgg Stage 0: Setting up Params")
    # do not count setup time
    total_time = total_time + timeit.default_timer()
    setup_param_results_and_failures = setup_param(
        clients=setup_param_clients,
        sec_agg_param_dict=sec_agg_param_dict
    )
    total_time = total_time - timeit.default_timer()
    # time consumption of stage 1 on the server side
    timer += [timeit.default_timer()]
    setup_param_results = setup_param_results_and_failures[0]
    ask_keys_clients: Dict[int, ClientProxy] = {}
    if len(setup_param_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after setup param stage")
    tmp_lst = [o[0] for o in setup_param_results]
    for idx, client in setup_param_clients.items():
        if client in tmp_lst:
            ask_keys_clients[idx] = client
    timer[-1] = timeit.default_timer() - timer[-1]

    # === Stage 1: Ask Public Keys ===
    log(INFO, "SecAgg Stage 1: Asking Keys")
    # exclude ask_keys
    total_time = total_time + timeit.default_timer()
    ask_keys_results_and_failures = ask_keys(ask_keys_clients)
    total_time = total_time - timeit.default_timer()
    timer += [timeit.default_timer()]
    public_keys_dict: Dict[int, AskKeysRes] = {}
    ask_keys_results = ask_keys_results_and_failures[0]
    if len(ask_keys_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after ask keys stage")
    share_keys_clients: Dict[int, ClientProxy] = {}

    # Build public keys dict
    # tmp_map = dict(ask_keys_results)
    for idx, client in ask_keys_clients.items():
        if client in [result[0] for result in ask_keys_results]:
            pos = [result[0] for result in ask_keys_results].index(client)
            public_keys_dict[idx] = ask_keys_results[pos][1]
            share_keys_clients[idx] = client
        # key = tmp_map.get(client, None)
        # if key is not None:
        #     public_keys_dict[idx] = key
        #     share_keys_clients[idx] = client
    timer[-1] = timeit.default_timer() - timer[-1]

    # === Stage 2: Share Keys ===
    log(INFO, "SecAgg Stage 2: Sharing Keys")
    total_time = total_time + timeit.default_timer()
    share_keys_results_and_failures = share_keys(
        share_keys_clients, public_keys_dict, sec_agg_param_dict[
            'sample_num'], sec_agg_param_dict['share_num']
    )
    total_time = total_time - timeit.default_timer()
    timer += [timeit.default_timer()]
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
    timer[-1] = timeit.default_timer() - timer[-1]

    # === Stage 3: Ask Vectors ===
    log(INFO, "SecAgg Stage 3: Asking Vectors")
    total_time = total_time + timeit.default_timer()
    ask_vectors_results_and_failures = ask_vectors(
        ask_vectors_clients, forward_packet_list_dict, client_instructions)
    total_time = total_time - timeit.default_timer()
    timer += [timeit.default_timer()]
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
    timer[-1] = timeit.default_timer() - timer[-1]
    # === Stage 4: Unmask Vectors ===
    shamir_reconstruction_time = 0
    prg_time = 0
    log(INFO, "SecAgg Stage 4: Unmasking Vectors")
    total_time = total_time + timeit.default_timer()
    unmask_vectors_results_and_failures = unmask_vectors(
        unmask_vectors_clients, dropout_clients, sec_agg_param_dict['sample_num'], sec_agg_param_dict['share_num'])
    unmask_vectors_results = unmask_vectors_results_and_failures[0]
    total_time = total_time - timeit.default_timer()
    timer += [timeit.default_timer()]
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
        shamir_reconstruction_time -= timeit.default_timer()
        secret = combine_shares(share_list)
        shamir_reconstruction_time += timeit.default_timer()
        if client_id in unmask_vectors_clients.keys():
            # seed is an available client's b
            prg_time -= timeit.default_timer()
            private_mask = sec_agg_primitives.pseudo_rand_gen(
                secret, sec_agg_param_dict['mod_range'], sec_agg_primitives.weights_shape(masked_vector))
            masked_vector = sec_agg_primitives.weights_subtraction(
                masked_vector, private_mask)
            prg_time += timeit.default_timer()
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
                prg_time -= timeit.default_timer()
                shared_key = sec_agg_primitives.generate_shared_key(
                    sec_agg_primitives.bytes_to_private_key(secret),
                    sec_agg_primitives.bytes_to_public_key(public_keys_dict[neighbor_id].pk1))
                pairwise_mask = sec_agg_primitives.pseudo_rand_gen(
                    shared_key, sec_agg_param_dict['mod_range'], sec_agg_primitives.weights_shape(masked_vector))
                prg_time += timeit.default_timer()
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
    aggregated_parameters = weights_to_parameters(aggregated_vector)
    timer[-1] = timeit.default_timer() - timer[-1]
    total_time = total_time + timeit.default_timer()
    f = open("log.txt", "a")
    f.write(f"Server time without communication:{total_time} \n")
    f.write(f"first element {aggregated_vector[0].flatten()[0]}\n\n\n")
    total_time = sum(timer)
    f.write("server time (detail):\n%s\n" %
            '\n'.join([f'stage {i} = {timer[i]} ({timer[i] * 100. / total_time:.2f} %)' for i in range(len(timer))]))
    f.write('shamir\'s key reconstruction time: %f (%.2f%%)\n' % (shamir_reconstruction_time,
                                                                  shamir_reconstruction_time * 100. / total_time))
    f.write('mask generation time: %f (%.2f%%)\n' % (prg_time, prg_time * 100. / total_time))
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
    clients: List[ClientProxy],
    sec_agg_param_dict: Dict[str, Scalar]
) -> SetupParamResultsAndFailures:
    def sec_agg_param_dict_with_sec_agg_id(sec_agg_param_dict: Dict[str, Scalar], sec_agg_id: int):
        new_sec_agg_param_dict = sec_agg_param_dict.copy()
        new_sec_agg_param_dict[
            'sec_agg_id'] = sec_agg_id
        return new_sec_agg_param_dict

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: setup_param_client(*p),
                (
                    c,
                    SetupParamIns(
                        sec_agg_param_dict=sec_agg_param_dict_with_sec_agg_id(
                            sec_agg_param_dict, idx),
                    ),
                ),
            )
            for idx, c in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, SetupParamRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def setup_param_client(client: ClientProxy, setup_param_msg: SetupParamIns) -> Tuple[ClientProxy, SetupParamRes]:
    setup_param_res = client.setup_param(setup_param_msg)
    return client, setup_param_res


def ask_keys(clients: List[ClientProxy]) -> AskKeysResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(ask_keys_client, c) for c in clients.values()]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, AskKeysRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def ask_keys_client(client: ClientProxy) -> Tuple[ClientProxy, AskKeysRes]:
    ask_keys_res = client.ask_keys(AskKeysIns())
    return client, ask_keys_res


def share_keys(clients: List[ClientProxy], public_keys_dict: Dict[int, AskKeysRes], sample_num: int,
               share_num: int) -> ShareKeysResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: share_keys_client(*p),
                (client, idx, public_keys_dict, sample_num, share_num),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, ShareKeysRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def share_keys_client(client: ClientProxy, idx: int, public_keys_dict: Dict[int, AskKeysRes], sample_num: int,
                      share_num: int) -> Tuple[ClientProxy, ShareKeysRes]:
    if share_num == sample_num:
        # complete graph
        return client, client.share_keys(ShareKeysIns(public_keys_dict=public_keys_dict))
    local_dict: Dict[int, AskKeysRes] = {}
    for i in range(-int(share_num / 2), int(share_num / 2) + 1):
        if ((i + idx) % sample_num) in public_keys_dict.keys():
            local_dict[(i + idx) % sample_num] = public_keys_dict[
                (i + idx) % sample_num
                ]

    return client, client.share_keys(ShareKeysIns(public_keys_dict=local_dict))


def ask_vectors(clients: List[ClientProxy], forward_packet_list_dict: Dict[int, List[ShareKeysPacket]],
                client_instructions: Dict[int, FitIns]) -> AskVectorsResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: ask_vectors_client(*p),
                (client, forward_packet_list_dict[idx], client_instructions[idx]),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, AskVectorsRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def ask_vectors_client(client: ClientProxy, forward_packet_list: List[ShareKeysPacket], fit_ins: FitIns) -> Tuple[
    ClientProxy, AskVectorsRes]:
    return client, client.ask_vectors(AskVectorsIns(ask_vectors_in_list=forward_packet_list, fit_ins=fit_ins))


def unmask_vectors(clients: List[ClientProxy], dropout_clients: List[ClientProxy], sample_num: int,
                   share_num: int) -> UnmaskVectorsResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: unmask_vectors_client(*p),
                (client, idx, list(clients.keys()), list(
                    dropout_clients.keys()), sample_num, share_num),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, UnmaskVectorsRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def unmask_vectors_client(client: ClientProxy, idx: int, clients: List[ClientProxy], dropout_clients: List[ClientProxy],
                          sample_num: int, share_num: int) -> Tuple[ClientProxy, UnmaskVectorsRes]:
    if share_num == sample_num:
        # complete graph
        return client, client.unmask_vectors(
            UnmaskVectorsIns(available_clients=clients, dropout_clients=dropout_clients))
    local_clients: List[int] = []
    local_dropout_clients: List[int] = []
    for i in range(-int(share_num / 2), int(share_num / 2) + 1):
        if ((i + idx) % sample_num) in clients:
            local_clients.append((i + idx) % sample_num)
        if ((i + idx) % sample_num) in dropout_clients:
            local_dropout_clients.append((i + idx) % sample_num)
    return client, client.unmask_vectors(
        UnmaskVectorsIns(available_clients=local_clients, dropout_clients=local_dropout_clients))
