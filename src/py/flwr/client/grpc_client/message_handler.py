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
"""Handle server messages by calling appropriate client methods."""


from typing import Tuple

from flwr.client.client import Client
from flwr.client.abc_sa_client_wrapper import SAClientWrapper
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage


# pylint: disable=missing-function-docstring


class UnkownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(
    client: Client, server_msg: ServerMessage
) -> Tuple[ClientMessage, int, bool]:
    if server_msg.HasField("reconnect"):
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect)
        return disconnect_msg, sleep_duration, False
    if server_msg.HasField("get_parameters"):
        return _get_parameters(client), 0, True
    if server_msg.HasField("fit_ins"):
        return _fit(client, server_msg.fit_ins), 0, True
    if server_msg.HasField("evaluate_ins"):
        return _evaluate(client, server_msg.evaluate_ins), 0, True
    if server_msg.HasField("sa_msg_carrier"):
        # f = open("logserver.txt", "a")
        # f.write(
        #     f"{server_msg.ByteSize()}\n")
        # f.close()
        t = (_sa_respond(client, server_msg.sa_msg_carrier), 0, True)
        # if client.id == 6:
        #     f = open("logclient.txt", "a")
        #     f.write(
        #         f"{t[0].ByteSize()}\n")
        #     f.close()
        return t
    if server_msg.HasField("sec_agg_msg"):
        f = open("logserver.txt", "a")
        f.write(
            f"{server_msg.ByteSize()}\n")
        f.close()
        if server_msg.sec_agg_msg.HasField("setup_param"):
            t = (_setup_param(client, server_msg.sec_agg_msg), 0, True)
            if client.sec_agg_id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.sec_agg_msg.HasField("ask_keys"):
            t = (_ask_keys(client, server_msg.sec_agg_msg), 0, True)
            if client.sec_agg_id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.sec_agg_msg.HasField("share_keys"):
            t = (_share_keys(client, server_msg.sec_agg_msg), 0, True)
            if client.sec_agg_id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.sec_agg_msg.HasField("ask_vectors"):
            t = (_ask_vectors(client, server_msg.sec_agg_msg), 0, True)
            if client.sec_agg_id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.sec_agg_msg.HasField("unmask_vectors"):
            t = (_unmask_vectors(client, server_msg.sec_agg_msg), 0, True)
            if client.sec_agg_id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
    if server_msg.HasField("light_sec_agg_ins"):
        f = open("logserver.txt", "a")
        f.write(
            f"{server_msg.ByteSize()}\n")
        f.close()
        if server_msg.light_sec_agg_ins.HasField("setup_cfg_ins"):
            t = (_light_sec_agg_setup_config(client, server_msg.light_sec_agg_ins), 0, True)
            if client.id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.light_sec_agg_ins.HasField("ask_en_msks_ins"):
            t = (_ask_encrypted_encoded_masks(client, server_msg.light_sec_agg_ins), 0, True)
            if client.id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.light_sec_agg_ins.HasField("ask_models_ins"):
            t = (_ask_masked_models(client, server_msg.light_sec_agg_ins), 0, True)
            if client.id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
        elif server_msg.light_sec_agg_ins.HasField("ask_agg_msks_ins"):
            t = (_ask_aggregated_encoded_masks(client, server_msg.light_sec_agg_ins), 0, True)
            if client.id == 3:
                f = open("logclient.txt", "a")
                f.write(
                    f"{t[0].ByteSize()}\n")
                f.close()
            return t
    raise UnkownServerMessage()


def _get_parameters(client: Client) -> ClientMessage:
    # No need to deserialize get_parameters_msg (it's empty)
    parameters_res = client.get_parameters()
    parameters_res_proto = serde.parameters_res_to_proto(parameters_res)
    return ClientMessage(parameters_res=parameters_res_proto)


def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)
    # Perform fit
    fit_res = client.fit(fit_ins)
    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)
    # Perform evaluation
    evaluate_res = client.evaluate(evaluate_ins)
    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)


def _reconnect(
    reconnect_msg: ServerMessage.Reconnect,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending Disconnect message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build Disconnect message
    disconnect = ClientMessage.Disconnect(reason=reason)
    return ClientMessage(disconnect=disconnect), sleep_duration


def _sa_respond(client: SAClientWrapper, msg: ServerMessage.SAMessageCarrier) -> ClientMessage:
    try:
        request = serde.sa_server_msg_carrier_from_proto(msg)
        response = client.sa_respond(request)
        response_msg = serde.sa_client_msg_carrier_to_proto(response)
        return ClientMessage(sa_msg_carrier=response_msg)
    except Exception as e:
        return _sa_error(msg, e)


def _setup_param(client: Client, setup_param_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        setup_param_ins = serde.setup_param_ins_from_proto(setup_param_msg)
        setup_param_res = client.setup_param(setup_param_ins)
        setup_param_res_proto = serde.setup_param_res_to_proto(setup_param_res)
        return ClientMessage(sec_agg_res=setup_param_res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_keys(client: Client, ask_keys_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        ask_keys_ins = serde.ask_keys_ins_from_proto(ask_keys_msg)
        ask_keys_res = client.ask_keys(ask_keys_ins)
        ask_keys_res_proto = serde.ask_keys_res_to_proto(ask_keys_res)
        return ClientMessage(sec_agg_res=ask_keys_res_proto)
    except Exception as e:
        return _error_res(e)


def _share_keys(client: Client, share_keys_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        share_keys_in = serde.share_keys_ins_from_proto(share_keys_msg)
        share_keys_res = client.share_keys(share_keys_in)
        share_keys_res_proto = serde.share_keys_res_to_proto(share_keys_res)
        return ClientMessage(sec_agg_res=share_keys_res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_vectors(client: Client, ask_vectors_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        ask_vectors_ins = serde.ask_vectors_ins_from_proto(ask_vectors_msg)
        ask_vectors_res = client.ask_vectors(ask_vectors_ins)
        ask_vectors_res_proto = serde.ask_vectors_res_to_proto(ask_vectors_res)
        return ClientMessage(sec_agg_res=ask_vectors_res_proto)
    except Exception as e:
        return _error_res(e)


def _unmask_vectors(client: Client, unmask_vectors_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        unmask_vectors_ins = serde.unmask_vectors_ins_from_proto(unmask_vectors_msg)
        unmask_vectors_res = client.unmask_vectors(
            unmask_vectors_ins)
        unmask_vectors_res_proto = serde.unmask_vectors_res_to_proto(unmask_vectors_res)
        return ClientMessage(sec_agg_res=unmask_vectors_res_proto)
    except Exception as e:
        return _error_res(e)


def _light_sec_agg_setup_config(client: Client, ins_proto: ServerMessage.LightSecAggIns) -> ClientMessage:
    try:
        ins = serde.light_sec_agg_setup_cfg_ins_from_proto(ins_proto)
        res = client.setup_config(ins)
        res_proto = serde.light_sec_agg_setup_cfg_res_to_proto(res)
        return ClientMessage(light_sec_agg_res=res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_encrypted_encoded_masks(client: Client, ins_proto: ServerMessage.LightSecAggIns) -> ClientMessage:
    try:
        ins = serde.ask_encrypted_encoded_masks_ins_from_proto(ins_proto)
        res = client.ask_encrypted_encoded_masks(ins)
        res_proto = serde.ask_encrypted_encoded_masks_res_to_proto(res)
        return ClientMessage(light_sec_agg_res=res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_masked_models(client: Client, ins_proto: ServerMessage.LightSecAggIns) -> ClientMessage:
    try:
        ins = serde.ask_masked_models_ins_from_proto(ins_proto)
        res = client.ask_masked_models(ins)
        res_proto = serde.ask_masked_models_res_to_proto(res)
        return ClientMessage(light_sec_agg_res=res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_aggregated_encoded_masks(client: Client, ins_proto: ServerMessage.LightSecAggIns) -> ClientMessage:
    try:
        ins = serde.ask_aggregated_encoded_masks_ins_from_proto(ins_proto)
        res = client.ask_aggregated_encoded_masks(ins)
        res_proto = serde.ask_aggregated_encoded_masks_res_to_proto(res)
        return ClientMessage(light_sec_agg_res=res_proto)
    except Exception as e:
        return _error_res(e)


def _error_res(e: Exception) -> ClientMessage:
    error_res = ClientMessage.SecAggRes(
        error_res=ClientMessage.SecAggRes.ErrorRes(error=e.args[0])
    )
    return ClientMessage(sec_agg_res=error_res)


def _sa_error(msg: ServerMessage.SAMessageCarrier, e: Exception) -> ClientMessage:
    res = ClientMessage.SAMessageCarrier(identifier=msg.identifier, error_msg=e.args[0])
    return ClientMessage(sa_msg_carrier=res)
