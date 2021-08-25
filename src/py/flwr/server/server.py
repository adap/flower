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
"""Flower server."""


import concurrent.futures
from pickle import LIST
import timeit
from logging import DEBUG, INFO, WARN, WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Reconnect,
    Scalar,
    Weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_weights
from flwr.common.secagg import secagg_primitives
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy, FedAvg
from flwr.common.typing import AskKeysIns, AskKeysRes, AskVectorsIns, AskVectorsRes, SetupParamIns, SetupParamRes, ShareKeysIns, ShareKeysPacket, ShareKeysRes, UnmaskVectorsIns, UnmaskVectorsRes
from flwr.server.strategy.secagg_strategy import SecAggStrategy

DEPRECATION_WARNING_EVALUATE = """
DEPRECATION WARNING: Method

    Server.evaluate(self, rnd: int) -> Optional[
        Tuple[Optional[float], EvaluateResultsAndFailures]
    ]

is deprecated and will be removed in a future release, use

    Server.evaluate_round(self, rnd: int) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]

instead.
"""

DEPRECATION_WARNING_EVALUATE_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_evaluate
return format:

    Strategy.aggregate_evaluate(...) -> Optional[float]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_evaluate(...) -> Tuple[Optional[float], Dict[str, Scalar]]

instead.
"""

DEPRECATION_WARNING_FIT_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_fit
return format:

    Strategy.aggregate_fit(...) -> Optional[Weights]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_fit(...) -> Tuple[Optional[Weights], Dict[str, Scalar]]

instead.
"""

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]

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


class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, secagg: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            if secagg == 1:
                # hard code methods
                if not isinstance(self.strategy, SecAggStrategy):
                    raise Exception("Strategy not compatible with secure aggregation")
                res_fit = self.sec_agg_fit_round(rnd=current_round)
                # TO BE REMOVED
                print(parameters_to_weights(res_fit[0]))
                raise Exception("SUCCESS")
                res_fit = self.fit_round(rnd=current_round)
                if res_fit:
                    parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                    log(INFO, parameters_prime)
                    if parameters_prime:
                        self.parameters = parameters_prime
            else:
                res_fit = self.fit_round(rnd=current_round)
                if res_fit:
                    parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                    if parameters_prime:
                        self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(rnd=current_round)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        rnd=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        log(WARNING, DEPRECATION_WARNING_EVALUATE)
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, _, results_and_failures = res
        return loss, results_and_failures

    def evaluate_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(client_instructions)
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],  # Deprecated
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result

        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(client_instructions)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        elif isinstance(aggregated_result, list):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = weights_to_parameters(aggregated_result)
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def sec_agg_fit_round(
        self,
        rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # Sample clients
        client_instruction_list = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager)
        setup_param_clients: Dict[int, ClientProxy] = {}
        client_instructions: Dict[int, FitIns] = {}
        for idx, value in enumerate(client_instruction_list):
            setup_param_clients[idx] = value[0]
            client_instructions[idx] = value[1]

        # Get sec agg parameters
        log(INFO, "SecAgg get param from dict")
        sec_agg_param_dict = self.strategy.get_sec_agg_param()
        sec_agg_param_dict["sample_num"] = len(client_instruction_list)
        sec_agg_param_dict = self.process_sec_agg_param_dict(sec_agg_param_dict)

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
        #masked_vector = secagg_primitives.weights_zero_generate(parameters_to_weights(self.parameters).shape)
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

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

    def _get_initial_parameters(self) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters

    def process_sec_agg_param_dict(self, sec_agg_param_dict: Dict[str, Scalar]) -> Dict[str, Scalar]:
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


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]]
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
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


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
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


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res


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
