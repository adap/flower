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
from logging import INFO
from flwr.common.sa_primitives.mpc_functions import LCC_decoding_with_points, model_unmasking
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.server import Server, ClientProxy
from flwr.common.typing import SAServerMessageCarrier
from flwr.common.sa_primitives import sec_agg_primitives
from flwr.common.timer import Timer
from flwr.common.secure_aggregation import SecureAggregationFitRound, SecureAggregationResultsAndFailures
from flwr.common.logger import log

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]


def padding(d, U, T):
    ret = d % (U - T)
    if ret != 0:
        ret = U - T - ret
    return d + ret


class LightSecAggFedAvg(SecureAggregationFitRound, FedAvg):
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
        def check_and_update(_in: SecureAggregationResultsAndFailures, stage_num):
            ret1 = _in[0]
            if len(ret1) < self.U:
                raise Exception(f"Insufficient active clients after Stage {stage_num}")
            ret2 = [c for c, res in ret1]
            return ret1, ret2

        tm = Timer()
        # sample clients
        tm.tic()
        ins_lst = self.configure_fit(rnd, server.parameters, server.client_manager())

        # Testing , to be removed================================================
        is_test = self.cfg_dict.setdefault('test', 0) == 1
        if is_test:
            vector_length = self.cfg_dict['test_vector_dimension']
        # End =================================================================
        else:
            init_weights = parameters_to_weights(server.parameters)
            vector_length = sum([o.size for o in init_weights])
        d = padding(vector_length + 1, self.U, self.T)

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

        # STAGE 0: setup config
        log(INFO, "LightSecAgg Stage 0: Setting up config")
        # exclude communication time
        # results_and_failures = self.setup_config(active_clients, self.cfg_dict)

        def new_dict(_in: dict, _id):
            ret = _in.copy()
            ret['id'] = _id
            return ret

        results_and_failures = self.sa_request([(k, SAServerMessageCarrier(
            identifier='0',
            str2scalar=new_dict(self.cfg_dict, v)
        )) for k, v in proxy2id.items()])

        tm.tic('s0')
        res_lst, active_clients = check_and_update(results_and_failures, 0)
        public_key_dict = dict([(str(proxy2id[c]), res.bytes_list[0]) for c, res in res_lst])
        tm.toc('s0')

        # STAGE 1: ask encrypted encoded sub-masks
        log(INFO, "LightSecAgg Stage 1: Ask encrypted encoded sub-masks")
        # results_and_failures = self.ask_encrypted_encoded_masks(active_clients, public_key_dict)
        results_and_failures = self.sa_request([
            (c, SAServerMessageCarrier('1', str2scalar=public_key_dict)) for c in active_clients])

        tm.tic('s1')
        res_lst, active_clients = check_and_update(results_and_failures, 1)
        # forward packets to their destinations
        fwd_packets_dict = dict([(proxy2id[c], dict()) for c in active_clients])
        for c, res in res_lst:
            for des, ctext in res.str2scalar.items():
                des = int(des)
                if des in fwd_packets_dict.keys():
                    fwd_packets_dict[des][str(proxy2id[c])] = ctext
        tm.toc('s1')

        # STAGE 2: ask masked models
        log(INFO, "LightSecAgg Stage 2: Ask masked models")
        # results_and_failures = self.ask_masked_models(active_clients, fwd_packets_dict, fit_ins_dict)
        results_and_failures = self.sa_request([
            (c, SAServerMessageCarrier('2', str2scalar=fwd_packets_dict[proxy2id[c]])) for c in active_clients])
        tm.tic('s2')
        res_lst, active_clients = check_and_update(results_and_failures, 2)
        # compute the aggregated masked model
        GF = self.GF
        agg_model = parameters_to_weights(res_lst[0][1].parameters)
        agg_model = [o.view(GF) for o in agg_model]
        for i in range(1, len(res_lst)):
            weights = parameters_to_weights(res_lst[i][1].parameters)
            weights = [o.view(GF) for o in weights]
            agg_model = sec_agg_primitives.weights_addition(agg_model, weights)
        tm.toc('s2')

        # STAGE 3: ask aggregated encoded sub-masks
        log(INFO, "LightSecAgg Stage 3: Ask aggregated encoded sub-masks")
        # results_and_failures = self.ask_aggregated_encoded_masks(active_clients)
        active_lst = np.array([proxy2id[c] for c in active_clients])
        results_and_failures = self.sa_request([
            (c, SAServerMessageCarrier('3', numpy_ndarray_list=[active_lst])) for c in active_clients])

        tm.tic('s3')
        res_lst, active_clients = check_and_update(results_and_failures, 3)
        # reconstruct mask
        alpha_s = np.arange(1, self.N + 1)
        beta_s = np.arange(self.N + 1, self.N + 1 + self.U)
        msk_buffer = np.zeros((self.U, d // (self.U - self.T)), dtype=np.int32).view(GF)
        active_ids = np.array([proxy2id[c] for c in active_clients])[:self.U]
        for i in range(self.U):
            tmp = parameters_to_weights(res_lst[i][1].parameters)[0]
            msk_buffer[i, :] = tmp
        agg_msk: np.ndarray = LCC_decoding_with_points(msk_buffer, alpha_s[active_ids], beta_s, GF)
        agg_msk = agg_msk.reshape(-1, 1)[:d]
        agg_model = model_unmasking(agg_model, agg_msk, GF)
        factor, agg_model = sec_agg_primitives.factor_weights_extract(agg_model)
        agg_model = sec_agg_primitives.weights_divide(agg_model, factor)
        # inverse quantize
        agg_model = sec_agg_primitives.reverse_quantize(agg_model, self.clipping_range, self.target_range)
        tm.toc('s3')
        tm.toc()
        times = tm.get_all()
        num_dropouts = self.N - len(active_clients)
        f = open("log.txt", "a")
        f.write(f"Server time with communication:{times['default']} \n")
        f.write(f"Server time without communication:{sum([times['s0'], times['s1'], times['s2'], times['s3']])} \n")
        f.write(f"first element {agg_model[0].flatten()[0]}\n\n\n")
        f.write("server time (detail):\n%s\n" %
                '\n'.join([f"round {i} = {times['s' + str(i)]} ({times['s' + str(i)] * 100. / times['default']:.2f} %)"
                           for i in range(4)]))
        f.write('shamir\'s key reconstruction time: %f (%.2f%%)\n' % (0.1, 0.1))
        f.write('mask generation time: %f (%.2f%%)\n' % (0.1, 0.1))
        f.write('num of dropouts: %d\n' % num_dropouts)
        f.close()
        return weights_to_parameters(agg_model), None, None




