from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple
from flwr.common.parameter import parameters_to_weights, weights_to_parameters
from flwr.common.typing import AskKeysIns, AskKeysRes, AskVectorsIns, AskVectorsRes, FitIns, FitRes, Parameters, Scalar, SetupParamIns, SetupParamRes, ShareKeysIns, ShareKeysPacket, ShareKeysRes, UnmaskVectorsIns, UnmaskVectorsRes
from flwr.server.client_proxy import ClientProxy
from flwr.common.secagg import secagg_primitives
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

    # Sample clients
    client_instruction_list = server.strategy.configure_fit(
        rnd=rnd, parameters=server.parameters, client_manager=server._client_manager)
    setup_param_clients: Dict[int, ClientProxy] = {}
    client_instructions: Dict[int, FitIns] = {}
    for idx, value in enumerate(client_instruction_list):
        setup_param_clients[idx] = value[0]
        client_instructions[idx] = value[1]

    # Get sec agg parameters
    log(INFO, "SecAgg get param from dict")
    sec_agg_param_dict = server.strategy.get_sec_agg_param()
    sec_agg_param_dict["sample_num"] = len(client_instruction_list)
    sec_agg_param_dict = process_sec_agg_param_dict(sec_agg_param_dict)

    # === Stage 0: Setup ===
    # Give rnd, sample_num, share_num, threshold, client id
    log(INFO, "SecAgg setup params")
    setup_param_results_and_failures = setup_param(
        clients=setup_param_clients,
        sec_agg_param_dict=sec_agg_param_dict
    )
    setup_param_results = setup_param_results_and_failures[0]
    ask_keys_clients: Dict[int, ClientProxy] = {}
    if len(setup_param_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after setup param stage")
    for idx, client in setup_param_clients.items():
        if client in [result[0] for result in setup_param_results]:
            ask_keys_clients[idx] = client

    # === Stage 1: Ask Public Keys ===
    log(INFO, "SecAgg ask keys")
    ask_keys_results_and_failures = ask_keys(ask_keys_clients)

    public_keys_dict: Dict[int, AskKeysRes] = {}
    ask_keys_results = ask_keys_results_and_failures[0]
    if len(ask_keys_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after ask keys stage")
    share_keys_clients: Dict[int, ClientProxy] = {}

    # Build public keys dict
    for idx, client in ask_keys_clients.items():
        if client in [result[0] for result in ask_keys_results]:
            pos = [result[0] for result in ask_keys_results].index(client)
            public_keys_dict[idx] = ask_keys_results[pos][1]
            share_keys_clients[idx] = client

    # === Stage 2: Share Keys ===
    log(INFO, "SecAgg share keys")
    share_keys_results_and_failures = share_keys(
        share_keys_clients, public_keys_dict, sec_agg_param_dict[
            'sample_num'], sec_agg_param_dict['share_num']
    )
    share_keys_results = share_keys_results_and_failures[0]
    if len(share_keys_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after share keys stage")

    # Build forward packet list dictionary
    total_packet_list: List[ShareKeysPacket] = []
    forward_packet_list_dict: Dict[int, List[ShareKeysPacket]] = {}
    ask_vectors_clients: Dict[int, ClientProxy] = {}
    for idx, client in share_keys_clients.items():
        if client in [result[0] for result in share_keys_results]:
            pos = [result[0] for result in share_keys_results].index(client)
            ask_vectors_clients[idx] = client
            packet_list = share_keys_results[pos][1].share_keys_res_list
            total_packet_list += packet_list

    for idx in ask_vectors_clients.keys():
        forward_packet_list_dict[idx] = []

    for packet in total_packet_list:
        destination = packet.destination
        if destination in ask_vectors_clients.keys():
            forward_packet_list_dict[destination].append(packet)

    # === Stage 3: Ask Vectors ===
    log(INFO, "SecAgg ask vectors")
    ask_vectors_results_and_failures = ask_vectors(
        ask_vectors_clients, forward_packet_list_dict, client_instructions)
    ask_vectors_results = ask_vectors_results_and_failures[0]
    if len(ask_vectors_results) < sec_agg_param_dict['min_num']:
        raise Exception("Not enough available clients after ask vectors stage")
    #masked_vector = secagg_primitives.weights_zero_generate(parameters_to_weights(server.parameters).shape)
    # testing code
    masked_vector = secagg_primitives.weights_zero_generate([(1), (2, 3), (2, 3)])
    # end testing code

    # Add all collected masked vectors and compuute available and dropout clients set
    unmask_vectors_clients: Dict[int, ClientProxy] = {}
    dropout_clients = ask_vectors_clients.copy()
    for idx, client in ask_vectors_clients.items():
        if client in [result[0] for result in ask_vectors_results]:
            pos = [result[0] for result in ask_vectors_results].index(client)
            unmask_vectors_clients[idx] = client
            dropout_clients.pop(idx)
            client_parameters = ask_vectors_results[pos][1].parameters
            masked_vector = secagg_primitives.weights_addition(
                masked_vector, parameters_to_weights(client_parameters))

    # === Stage 4: Unmask Vectors ===
    log(INFO, "SecAgg unmask vectors")
    unmask_vectors_results_and_failures = unmask_vectors(
        unmask_vectors_clients, dropout_clients, sec_agg_param_dict['sample_num'], sec_agg_param_dict['share_num'])
    unmask_vectors_results = unmask_vectors_results_and_failures[0]

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
    # but divide vector by number of clients available after ask vectors
    for client_id, share_list in collected_shares_dict.items():
        if len(share_list) < sec_agg_param_dict['threshold']:
            raise Exception(
                "Not enough shares to recover secret in unmask vectors stage")
        seed = secagg_primitives.combine_shares(share_list=share_list)
        if client_id in unmask_vectors_clients.keys():
            # seed is an available client's b
            private_mask = secagg_primitives.pseudo_rand_gen(
                seed, sec_agg_param_dict['mod_range'], secagg_primitives.weights_shape(masked_vector))
            masked_vector = secagg_primitives.weights_subtraction(
                masked_vector, private_mask)
        else:
            # seed is a dropout client's sk1
            neighbor_list: List[int] = []
            if sec_agg_param_dict['share_num'] == sec_agg_param_dict['sample_num']:
                neighbor_list = list(ask_vectors_clients.keys()).remove(client_id)
            else:
                for i in range(-int(sec_agg_param_dict['share_num'] / 2), int(sec_agg_param_dict['share_num'] / 2) + 1):
                    if i != 0 and ((i + client_id) % sec_agg_param_dict['sample_num']) in ask_vectors_clients.keys():
                        neighbor_list.append((i + client_id) %
                                             sec_agg_param_dict['sample_num'])
            for neighbor_id in neighbor_list:
                shared_key = secagg_primitives.generate_shared_key(
                    seed, secagg_primitives.bytes_to_public_key(public_keys_dict[neighbor_id].pk1))
                pairwise_mask = secagg_primitives.pseudo_rand_gen(
                    shared_key, sec_agg_param_dict['mod_range'], secagg_primitives.weights_shape(masked_vector))
                if client_id > neighbor_id:
                    masked_vector = secagg_primitives.weights_addition(
                        masked_vector, pairwise_mask)
                else:
                    masked_vector = secagg_primitives.weights_subtraction(
                        masked_vector, pairwise_mask)
    masked_vector = secagg_primitives.weights_mod(
        masked_vector, sec_agg_param_dict['mod_range'])
    # Divide vector by number of clients who have given us their masked vector
    # i.e. those participating in final unmask vectors stage
    total_weights_factor, masked_vector = secagg_primitives.factor_weights_extract(
        masked_vector)
    masked_vector = secagg_primitives.weights_divide(
        masked_vector, total_weights_factor)
    aggregated_vector = secagg_primitives.reverse_quantize(
        masked_vector, sec_agg_param_dict['clipping_range'], sec_agg_param_dict['target_range'])
    aggregated_parameters = weights_to_parameters(aggregated_vector)
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
                2, int(0.9*sec_agg_param_dict['sample_num']))
    else:
        if 'min_num' not in sec_agg_param_dict:
            sec_agg_param_dict['min_num'] = int(
                sec_agg_param_dict['min_frac']*sec_agg_param_dict['sample_num'])
        else:
            sec_agg_param_dict['min_num'] = max(sec_agg_param_dict['min_num'], int(
                sec_agg_param_dict['min_frac']*sec_agg_param_dict['sample_num']))

    if 'share_num' not in sec_agg_param_dict:
        # Complete graph
        sec_agg_param_dict['share_num'] = sec_agg_param_dict['sample_num']
    elif sec_agg_param_dict['share_num'] % 2 == 0 and sec_agg_param_dict['share_num'] != sec_agg_param_dict['sample_num']:
        # we want share_num of each node to be either odd or sample_num
        log(WARNING, "share_num value changed due to sample num and share_num constraints! See documentation for reason")
        sec_agg_param_dict['share_num'] += 1

    if 'threshold' not in sec_agg_param_dict:
        sec_agg_param_dict['threshold'] = max(
            2, int(sec_agg_param_dict['share_num'] * 0.9))

    # To be modified
    if 'max_weights_factor' not in sec_agg_param_dict:
        sec_agg_param_dict['max_weights_factor'] = 5

    # Quantization parameters
    if 'clipping_range' not in sec_agg_param_dict:
        sec_agg_param_dict['clipping_range'] = 3

    if 'target_range' not in sec_agg_param_dict:
        sec_agg_param_dict['target_range'] = 10000

    if 'mod_range' not in sec_agg_param_dict:
        sec_agg_param_dict['mod_range'] = sec_agg_param_dict['sample_num'] * \
            sec_agg_param_dict['target_range'] * \
            sec_agg_param_dict['max_weights_factor']

    if 'timeout' not in sec_agg_param_dict:
        sec_agg_param_dict['timeout'] = 20

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
        and (sec_agg_param_dict['share_num'] % 2 == 1 or sec_agg_param_dict['share_num'] == sec_agg_param_dict['sample_num'])
        and sec_agg_param_dict['target_range']*sec_agg_param_dict['sample_num']*sec_agg_param_dict['max_weights_factor'] <= sec_agg_param_dict['mod_range']
    ), "SecAgg parameters not accepted"
    return sec_agg_param_dict


def setup_param(
    clients: List[ClientProxy],
    sec_agg_param_dict: Dict[str, Scalar]
) -> SetupParamResultsAndFailures:
    def sec_agg_param_dict_with_sec_agg_id(sec_agg_param_dict: Dict[str, Scalar], secagg_id: int):
        new_sec_agg_param_dict = sec_agg_param_dict.copy()
        new_sec_agg_param_dict[
            'secagg_id'] = secagg_id
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
    print(failures)
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


def share_keys(clients: List[ClientProxy], public_keys_dict: Dict[int, AskKeysRes], sample_num: int, share_num: int) -> ShareKeysResultsAndFailures:
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


def share_keys_client(client: ClientProxy, idx: int, public_keys_dict: Dict[int, AskKeysRes], sample_num: int, share_num: int) -> Tuple[ClientProxy, ShareKeysRes]:
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


def ask_vectors(clients: List[ClientProxy], forward_packet_list_dict: Dict[int, List[ShareKeysPacket]], client_instructions: Dict[int, FitIns]) -> AskVectorsResultsAndFailures:
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


def ask_vectors_client(client: ClientProxy, forward_packet_list: List[ShareKeysPacket], fit_ins: FitIns) -> Tuple[ClientProxy, AskVectorsRes]:

    return client, client.ask_vectors(AskVectorsIns(ask_vectors_in_list=forward_packet_list, fit_ins=fit_ins))


def unmask_vectors(clients: List[ClientProxy], dropout_clients: List[ClientProxy], sample_num: int, share_num: int) -> UnmaskVectorsResultsAndFailures:
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


def unmask_vectors_client(client: ClientProxy, idx: int, clients: List[ClientProxy], dropout_clients: List[ClientProxy], sample_num: int, share_num: int) -> Tuple[ClientProxy, UnmaskVectorsRes]:
    if share_num == sample_num:
        # complete graph
        return client, client.unmask_vectors(UnmaskVectorsIns(available_clients=clients, dropout_clients=dropout_clients))
    local_clients: List[int] = []
    local_dropout_clients: List[int] = []
    for i in range(-int(share_num / 2), int(share_num / 2) + 1):
        if ((i + idx) % sample_num) in clients:
            local_clients.append([(i + idx) % sample_num])
        if ((i + idx) % sample_num) in dropout_clients:
            local_dropout_clients.append([(i + idx) % sample_num])
    return client, client.unmask_vectors(UnmaskVectorsIns(available_clients=local_clients, dropout_clients=dropout_clients))
