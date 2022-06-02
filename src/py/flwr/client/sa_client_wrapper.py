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
from abc import ABC, abstractmethod
from flwr.common import (
    AskKeysRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,

)
from typing import Dict
from flwr.common.typing import AskKeysIns, AskVectorsIns, AskVectorsRes, SetupParamIns, ShareKeysIns, ShareKeysRes, \
    UnmaskVectorsIns, UnmaskVectorsRes, LightSecAggSetupConfigIns, LightSecAggSetupConfigRes, AskEncryptedEncodedMasksIns, \
    AskEncryptedEncodedMasksRes, EncryptedEncodedMasksPacket, Parameters, AskMaskedModelsIns, AskMaskedModelsRes, \
    AskAggregatedEncodedMasksIns, AskAggregatedEncodedMasksRes, SAServerMessageCarrier, SAClientMessageCarrier, \
    ShareKeysPacket
from flwr.common.sec_agg import sec_agg_client_logic
from flwr.common.light_sec_agg import client_logic as lsa_proto
import numpy as np
from .client import Client
import sys
from flwr.common.timer import Timer


class SAClient(Client, ABC):
    """Wrapper which adds SecAgg methods."""

    def __init__(self, c: Client) -> None:
        self.client = c

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        return self.client.get_parameters()

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    @abstractmethod
    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        """response to the server for Secure Aggregation"""


class LightSecAggWrapper(SAClient):
    """Wrapper which adds LightSecAgg methods."""
    def __init__(self, c: Client):
        super().__init__(c)
        self.tm = Timer()

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        self.tm.tic('s' + ins.identifier)
        if ins.identifier == '0':
            new_ins = LightSecAggSetupConfigIns(ins.str2scalar)
            res = lsa_proto.setup_config(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='0', bytes_list=[res.pk])
        elif ins.identifier == '1':
            public_keys_dict = dict([(int(k), LightSecAggSetupConfigRes(v)) for k, v in ins.str2scalar.items()])
            new_ins = AskEncryptedEncodedMasksIns(public_keys_dict)
            res = lsa_proto.ask_encrypted_encoded_masks(self, new_ins)
            packet_dict = dict([(str(p.destination), p.ciphertext) for p in res.packet_list])
            ret_msg = SAClientMessageCarrier(identifier='1', str2scalar=packet_dict)
        elif ins.identifier == '2':
            packets = [EncryptedEncodedMasksPacket(int(k), self.id, v) for k, v in ins.str2scalar.items()]
            new_ins = AskMaskedModelsIns(packets, ins.fit_ins)
            res = lsa_proto.ask_masked_models(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='2', parameters=res.parameters)
        elif ins.identifier == '3':
            new_ins = AskAggregatedEncodedMasksIns(ins.numpy_ndarray_list[0].tolist())
            res = lsa_proto.ask_aggregated_encoded_masks(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='3', parameters=res.aggregated_encoded_mask)
        self.tm.toc('s' + ins.identifier)
        if self.id == 6:
            f = open("log.txt", "a")
            f.write(f"Client without communication stage {ins.identifier}:{self.tm.get('s' + ins.identifier)} \n")
            f.close()

        return ret_msg


class SecAggPlusWrapper(SAClient):

    def __init__(self, c: Client):
        super().__init__(c)
        self.tm = Timer()

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        self.tm.tic('s' + ins.identifier)
        if ins.identifier == '0':
            sec_agg_client_logic.setup_param(self, SetupParamIns(ins.str2scalar))
            res = sec_agg_client_logic.ask_keys(self, AskKeysIns())
            ret_msg = SAClientMessageCarrier('1', bytes_list=[res.pk1, res.pk2])
        # if ins.identifier == '1':
        #     res = sec_agg_client_logic.ask_keys(self, AskKeysIns())
        #     return SAClientMessageCarrier('1', bytes_list=[res.pk1, res.pk2])
        elif ins.identifier == '1':
            new_dict: Dict[int, AskKeysRes] = {}
            # key of received dict is like 'idx_pk1' or 'idx_pk2'
            for k, pk in ins.str2scalar.items():
                idx, token = k.split('_')
                idx = int(idx)
                t = new_dict.setdefault(idx, AskKeysRes(b'', b''))
                if token == 'pk1':
                    t.pk1 = pk
                else:
                    t.pk2 = pk
            res = sec_agg_client_logic.share_keys(self, ShareKeysIns(new_dict))
            source_lst = np.array([o.source for o in res.share_keys_res_list])
            destination_lst = np.array([o.destination for o in res.share_keys_res_list])
            ciphertext_lst = [o.ciphertext for o in res.share_keys_res_list]
            ret_msg = SAClientMessageCarrier('1', numpy_ndarray_list=[source_lst, destination_lst],
                                          bytes_list=ciphertext_lst)
        elif ins.identifier == '2':
            src_lst = ins.numpy_ndarray_list[0]
            des_lst = ins.numpy_ndarray_list[1]
            txt_lst = ins.bytes_list
            packet_lst = [ShareKeysPacket(s, d, t) for s, d, t in zip(src_lst, des_lst, txt_lst)]
            res = sec_agg_client_logic.ask_vectors(self, AskVectorsIns(packet_lst, ins.fit_ins))
            ret_msg = SAClientMessageCarrier('2', parameters=res.parameters)
        elif ins.identifier == '3':
            actives, dropouts = ins.numpy_ndarray_list[0], ins.numpy_ndarray_list[1]
            actives, dropouts = actives.tolist(), dropouts.tolist()
            res = sec_agg_client_logic.unmask_vectors(self, UnmaskVectorsIns(actives, dropouts))
            ret_msg = SAClientMessageCarrier('3', str2scalar=dict([(str(k), v) for k, v in res.share_dict.items()]))
        self.tm.toc('s' + ins.identifier)
        if self.sec_agg_id == 6:
            f = open("log.txt", "a")
            f.write(f"Client without communication stage {ins.identifier}:{self.tm.get('s' + ins.identifier)} \n")
            if ins.identifier == '3':
                times = self.tm.get_all()
                f.write(f"Client without communication total: {sum([times['s0'], times['s1'], times['s2'], times['s3']])} \n")
            f.close()

        return ret_msg

    # def setup_param(self, setup_param_ins: SetupParamIns):
    #     return sec_agg_client_logic.setup_param(self, setup_param_ins)
    #
    # def ask_keys(self, ask_keys_ins: AskKeysIns) -> AskKeysRes:
    #     return sec_agg_client_logic.ask_keys(self, ask_keys_ins)
    #
    # def share_keys(self, share_keys_in: ShareKeysIns) -> ShareKeysRes:
    #     return sec_agg_client_logic.share_keys(self, share_keys_in)
    #
    # def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
    #     return sec_agg_client_logic.ask_vectors(self, ask_vectors_ins)
