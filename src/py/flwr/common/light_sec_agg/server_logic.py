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
from logging import INFO, WARNING
import math
from mpc_functions import LCC_decoding_with_points, model_unmasking
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.server import Server
from flwr.common.light_sec_agg.protocol import AskAggregatedEncodedMasksResultsAndFailures, FitResultsAndFailures
from flwr.common.typing import LightSecAggSetupConfigIns, AskEncryptedEncodedMasksIns, AskMaskedModelsIns, \
    AskAggregatedEncodedMasksIns, LightSecAggSetupConfigRes, AskEncryptedEncodedMasksRes, AskMaskedModelsRes, \
    AskAggregatedEncodedMasksRes, EncryptedEncodedMasksPacket
from flwr.server.client_proxy import ClientProxy
from flwr.client.sec_agg_client import LightSecAggClient
from flwr.common.sec_agg import sec_agg_primitives
import timeit
import sys
from protocol import LightSecAggProtocol, SecureAggregationFitRound, LightSecAggSetupConfigResultsAndFailures, \
    AskEncryptedEncodedMasksResultsAndFailures, AskMaskedModelsResultsAndFailures
import concurrent.futures

from flwr.common.logger import log

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)


def padding(d, U, T):
    ret = d % (U - T)
    if ret != 0:
        ret = U - T - ret
    return d + ret


class LightSecAgg(LightSecAggProtocol, SecureAggregationFitRound, FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        cfg_dict: Dict[str, Scalar] = None
    ) -> None:
        FedAvg.__init__(self, fraction_fit=fraction_fit,
                        fraction_eval=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_eval_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        accept_failures=accept_failures,
                        initial_parameters=initial_parameters)
        assert {'sample_num', 'privacy_guarantee', 'min_clients', 'prime_number', 'clipping_range',
                'target_range'}.issubset(cfg_dict.keys()), "Necessary keys unfounded in the configuration dictionary."
        self.N = cfg_dict['sample_num']
        self.T = cfg_dict['privacy_guarantee']
        self.U = cfg_dict['min_clients']
        self.p = cfg_dict['prime_number']
        self.clipping_range = cfg_dict['clipping_range']
        self.target_range = cfg_dict['target_range']
        assert self.target_range < self.p, "Target range must be smaller than the prime number"
        if 'max_weights_factor' not in cfg_dict.keys():
            cfg_dict['max_weights_factor'] = self.p // self.target_range
        self.max_weights_factor = cfg_dict['max_weights_factor']
        self.cfg_dict = cfg_dict
        assert self.N >= self.U > self.T >= 0
        self.GF = galois.GF(self.p)
        self.proxy2id = {}

    def fit_round(self, server: Server, rnd: int) -> Optional[Tuple[Optional[Parameters],
                                                                    Dict[str, Scalar],
                                                                    FitResultsAndFailures]]:
        stopwatch = []

        def tic():
            stopwatch.append(-timeit.default_timer())

        def toc():
            stopwatch[-1] += timeit.default_timer()

        def check_and_update(_in, stage_num):
            ret1 = _in[0]
            if len(ret1) < self.U:
                raise Exception(f"Insufficient active clients after Stage {stage_num}")
            ret2 = [c for c, res in ret1]
            return ret1, ret2

        # sample clients
        tic()
        ins_lst = self.configure_fit(rnd, server.parameters, server.client_manager())

        # Testing , to be removed================================================
        is_test = self.cfg_dict.setdefault('test', 0) == 1
        if is_test:
            vector_length = self.cfg_dict['test_vector_dimension']
            d = padding(vector_length, self.U, self.T)
        # End =================================================================
        else:
            init_weights = parameters_to_weights(server.parameters)
            vector_length = sum([o.size for o in init_weights])
            d = padding(vector_length, self.U, self.T)

        active_clients = []
        # map ClientProxy to Secure Aggregation ID
        proxy2id = {}
        # map Secure Aggregation ID to fit instructions
        fit_ins_dict = {}
        for i, o in enumerate(ins_lst):
            active_clients.append(o[0])
            proxy2id[o[0]] = i
            fit_ins_dict[i] = o[1]
        assert self.N == len(ins_lst)
        self.proxy2id = proxy2id
        toc()

        # STAGE 0: setup config
        log(INFO, "LightSecAgg Stage 0: Setting up config")
        # exclude communication time
        results_and_failures = self.setup_config(active_clients, self.cfg_dict)
        tic()
        res_lst, active_clients = check_and_update(results_and_failures, 0)
        public_key_dict = dict([(proxy2id[c], res) for c, res in res_lst])

        toc()

        # STAGE 1: ask encrypted encoded sub-masks
        log(INFO, "LightSecAgg Stage 1: Ask encrypted encoded sub-masks")
        results_and_failures = self.ask_encrypted_encoded_masks(active_clients, public_key_dict)
        tic()
        res_lst, active_clients = check_and_update(results_and_failures, 1)
        res_lst: List[Tuple[ClientProxy, AskEncryptedEncodedMasksRes]] = res_lst
        # forward packets to their destinations
        fwd_packets_dict = dict([(proxy2id[c], []) for c in active_clients])
        for c, res in res_lst:
            for packet in res.packet_list:
                if packet.destination in fwd_packets_dict.keys():
                    fwd_packets_dict[packet.destination].append(packet)
        toc()

        # STAGE 2: ask masked models
        log(INFO, "LightSecAgg Stage 2: Ask masked models")
        results_and_failures = self.ask_masked_models(active_clients, fwd_packets_dict, fit_ins_dict)
        tic()
        res_lst, active_clients = check_and_update(results_and_failures, 2)
        res_lst: List[Tuple[ClientProxy, AskMaskedModelsRes]] = res_lst
        # compute the aggregated masked model
        GF = self.GF
        agg_model = parameters_to_weights(res_lst[0][1].parameters)
        agg_model = [o.view(GF) for o in agg_model]
        for i in range(1, len(res_lst)):
            weights = parameters_to_weights(res_lst[i][1].parameters)
            weights = [o.view(GF) for o in weights]
            agg_model = sec_agg_primitives.weights_addition(agg_model, weights)
        toc()

        # STAGE 3: ask aggregated encoded sub-masks
        log(INFO, "LightSecAgg Stage 3: Ask aggregated encoded sub-masks")
        results_and_failures = self.ask_masked_models(active_clients, fwd_packets_dict, fit_ins_dict)
        tic()
        res_lst, active_clients = check_and_update(results_and_failures, 3)
        res_lst: List[Tuple[ClientProxy, AskAggregatedEncodedMasksRes]] = res_lst
        # reconstruct mask
        alpha_s = np.arange(1, self.N + 1)
        beta_s = np.arange(self.N + 1, self.N + 1 + self.U)
        msk_buffer = np.zeros((self.U, d // (self.U - self.T)), dtype=np.int32).view(GF)
        active_ids = np.array([proxy2id[c] for c in active_clients])
        for i in range(self.U):
            tmp = parameters_to_weights(res_lst[i][1].aggregated_encoded_mask)[0]
            msk_buffer[i, :] = tmp
        agg_msk: np.ndarray = LCC_decoding_with_points(msk_buffer, alpha_s[active_ids], beta_s, GF)
        agg_msk = agg_msk.reshape(-1, 1)[:d]
        agg_model = model_unmasking(agg_model, agg_msk, GF)
        # inverse quantize
        agg_model = sec_agg_primitives.reverse_quantize(agg_model, self.clipping_range, self.target_range)
        toc()
        return weights_to_parameters(agg_model), None, None

    def setup_config(self, clients: List[ClientProxy],
                     config_dict: Dict[str, Scalar]) -> LightSecAggSetupConfigResultsAndFailures:

        def get_ins(idx):
            ret = config_dict.copy()
            ret['id'] = idx
            return LightSecAggSetupConfigIns(ret)

        ins_lst = [get_ins(self.proxy2id[c]) for c in clients]
        return parallel(client_setup_config, clients, ins_lst)

    def ask_encrypted_encoded_masks(self, clients: List[ClientProxy],
                                    public_keys_dict: Dict[int, LightSecAggSetupConfigRes]
                                    ) -> AskEncryptedEncodedMasksResultsAndFailures:
        ins_lst = [AskEncryptedEncodedMasksIns(public_keys_dict)] * len(clients)
        return parallel(client_ask_encrypted_encoded_masks, clients, ins_lst)

    def ask_masked_models(self, clients: List[ClientProxy],
                          forward_packet_list_dict: Dict[int, List[EncryptedEncodedMasksPacket]],
                          client_instructions: Dict[int, FitIns] = None) -> AskMaskedModelsResultsAndFailures:
        ids = [self.proxy2id[c] for c in clients]
        ins_lst = [AskMaskedModelsIns(
            packet_list=forward_packet_list_dict[idx],
            fit_ins=client_instructions[idx] if client_instructions is not None else None
        ) for idx in ids]
        return parallel(client_ask_masked_models, clients, ins_lst)

    def ask_aggregated_encoded_masks(self, clients: List[ClientProxy]) -> AskAggregatedEncodedMasksResultsAndFailures:
        ids = [self.proxy2id[c] for c in clients]
        ins_lst = [AskAggregatedEncodedMasksIns(ids)] * len(clients)
        return parallel(client_ask_aggregated_encoded_masks, clients, ins_lst)


def client_setup_config(client: LightSecAggClient, ins: LightSecAggSetupConfigIns):
    return client, client.setup_config(ins)


def client_ask_encrypted_encoded_masks(client: LightSecAggClient, ins: AskEncryptedEncodedMasksIns):
    return client, client.ask_encrypted_encoded_masks(ins)


def client_ask_masked_models(client: LightSecAggClient, ins: AskMaskedModelsIns):
    return client, client.ask_masked_models(ins)


def client_ask_aggregated_encoded_masks(client: LightSecAggClient, ins: AskAggregatedEncodedMasksIns):
    return client, client.ask_aggregated_encoded_masks(ins)


def parallel(fn, clients, ins_lst):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda p: fn(*p), (client, ins))
            for client, ins in zip(clients, ins_lst)
        ]
        concurrent.futures.wait(futures)
    results = []
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




