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
"""(De-)Serialization Tests."""

from typing import Union, cast
import flwr.common.typing as ft
from .serde import scalar_from_proto, scalar_to_proto, light_sec_agg_setup_cfg_res_to_proto, \
    light_sec_agg_setup_cfg_res_from_proto, light_sec_agg_setup_cfg_ins_to_proto, light_sec_agg_setup_cfg_ins_from_proto \
    , ask_encrypted_encoded_masks_ins_to_proto, ask_encrypted_encoded_masks_ins_from_proto, \
    ask_masked_models_ins_from_proto, ask_masked_models_ins_to_proto, ask_aggregated_encoded_masks_ins_to_proto, \
    ask_aggregated_encoded_masks_ins_from_proto, ask_aggregated_encoded_masks_res_to_proto, \
    ask_encrypted_encoded_masks_res_to_proto, ask_masked_models_res_to_proto, ask_masked_models_res_from_proto, \
    ask_encrypted_encoded_masks_res_from_proto, ask_aggregated_encoded_masks_res_from_proto


def test_serialisation_deserialisation() -> None:
    """Test if after serialization/deserialisation the np.ndarray is
    identical."""

    # Prepare
    scalars = [True, b"bytestr", 3.14, 9000, "Hello"]

    for scalar in scalars:
        # Execute
        scalar = cast(Union[bool, bytes, float, int, str], scalar)
        serialized = scalar_to_proto(scalar)
        actual = scalar_from_proto(serialized)

        # Assert
        assert actual == scalar


def test_light_sec_agg():
    # stage 0
    inputs = [{'aa bb': 1.5, 'ccs': 1 << 20}]
    for o in inputs:
        serialized = light_sec_agg_setup_cfg_ins_to_proto(ft.LightSecAggSetupConfigIns(o))
        actual = light_sec_agg_setup_cfg_ins_from_proto(serialized).sec_agg_cfg_dict
        assert actual == o

    inputs = [b'aldshfads', b'3121jlfdsa  sa']
    for o in inputs:
        serialized = light_sec_agg_setup_cfg_res_to_proto(ft.LightSecAggSetupConfigRes(o))
        actual = light_sec_agg_setup_cfg_res_from_proto(serialized).pk
        assert o == actual

    # stage 1
    plst = [ft.EncryptedEncodedMasksPacket(1, 2, b'ttfftt'),
            ft.EncryptedEncodedMasksPacket(12, 998, b'%^&*ggggggfffff')]

    def check_plst(lst1, lst2):
        assert len(lst1) == len(lst2)
        for o1, o2 in zip(lst1, lst2):
            assert o1.source == o2.source and o1.destination == o2.destination
            assert o1.ciphertext == o2.ciphertext

    tmp_fn = ft.LightSecAggSetupConfigRes
    inputs = [{1: tmp_fn(b'hahahaha'), 2: tmp_fn(b'any !%#*@) text'), 3: tmp_fn(b'balala neng liang')}]
    for o in inputs:
        serialized = ask_encrypted_encoded_masks_ins_to_proto(ft.AskEncryptedEncodedMasksIns(o))
        actual = ask_encrypted_encoded_masks_ins_from_proto(serialized).public_keys_dict
        assert len(o) == len(actual)
        for k, v in o.items():
            assert actual[k].pk == v.pk

    inputs = [plst]
    for o in inputs:
        serialized = ask_encrypted_encoded_masks_res_to_proto(ft.AskEncryptedEncodedMasksRes(plst))
        actual = ask_encrypted_encoded_masks_res_from_proto(serialized)
        check_plst(o, actual.packet_list)

    # stage 2

    inputs = [ft.AskMaskedModelsIns(
        packet_list=plst,
        fit_ins=ft.FitIns(ft.Parameters([b'params'], 'my_type'), {'aa bb': 1.5, 'ccs': 1 << 20})
    )]
    for o in inputs:
        serialized = ask_masked_models_ins_to_proto(o)
        actual = ask_masked_models_ins_from_proto(serialized)
        check_plst(o.packet_list, actual.packet_list)
        assert o.fit_ins.parameters.tensors == actual.fit_ins.parameters.tensors
        assert o.fit_ins.parameters.tensor_type == actual.fit_ins.parameters.tensor_type
        assert o.fit_ins.config == actual.fit_ins.config

    inputs = [ft.Parameters([b'1', b'2', b'3', b'acb'], 'tt1')]
    for o in inputs:
        serialized = ask_masked_models_res_to_proto(ft.AskMaskedModelsRes(o))
        actual = ask_masked_models_res_from_proto(serialized)
        assert o.tensors == actual.parameters.tensors
        assert o.tensor_type == actual.parameters.tensor_type

    # stage 3
    inputs = [ft.AskAggregatedEncodedMasksIns([1, 2, 43215, 32144, 10])]
    for o in inputs:
        serialized = ask_aggregated_encoded_masks_ins_to_proto(o)
        actual = ask_aggregated_encoded_masks_ins_from_proto(serialized)
        assert o.surviving_clients == actual.surviving_clients

    inputs = [ft.Parameters([b'a', b'b', b'c', b'023189'], 'tt1')]
    for o in inputs:
        serialized = ask_aggregated_encoded_masks_res_to_proto(ft.AskAggregatedEncodedMasksRes(o))
        actual = ask_aggregated_encoded_masks_res_from_proto(serialized)
        assert o.tensors == actual.aggregated_encoded_mask.tensors
        assert o.tensor_type == actual.aggregated_encoded_mask.tensor_type
