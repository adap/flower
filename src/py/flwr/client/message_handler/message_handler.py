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
"""Client-side message handler."""


from typing import Tuple

from flwr.client.client import (
    Client,
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage


class UnknownServerMessage(Exception):
    """Exception indicating that the received message is unknown."""


def handle(
    client: Client, server_msg: ServerMessage
) -> Tuple[ClientMessage, int, bool]:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client : Client
        The Client instance provided by the user.
    server_msg: ServerMessage
        The message coming from the server, to be processed by the client.

    Returns
    -------
    client_msg: ClientMessage
        The result message that should be returned to the server.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the client should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    field = server_msg.WhichOneof("msg")
    if field == "reconnect_ins":
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
        return disconnect_msg, sleep_duration, False
    if field == "get_properties_ins":
        return _get_properties(client, server_msg.get_properties_ins), 0, True
    if field == "get_parameters_ins":
        return _get_parameters(client, server_msg.get_parameters_ins), 0, True
    if field == "fit_ins":
        return _fit(client, server_msg.fit_ins), 0, True
    if field == "evaluate_ins":
        return _evaluate(client, server_msg.evaluate_ins), 0, True
    if server_msg.HasField("example_ins"):
        return _example_response(client, server_msg.example_ins), 0, True
    if server_msg.HasField("send_vector_a_ins"):
        return _distribute_vector_a(client, server_msg.send_vector_a_ins), 0, True
    if server_msg.HasField("send_allpub_ins"):
        return _distribute_aggregated_pubkey(client, server_msg.send_allpub_ins), 0, True
    if server_msg.HasField("request_encrypted_ins"):
        return _request_encrypted_parameters(client, server_msg.request_encrypted_ins), 0, True
    if server_msg.HasField("send_csum_ins"):
        return _request_decryption_shares(client, server_msg.send_csum_ins), 0, True
    if server_msg.HasField("send_new_weights_ins"):
        return _request_modelupdate_confirmation(client, server_msg.send_new_weights_ins), 0, True
    raise UnknownServerMessage()


def _reconnect(
    reconnect_msg: ServerMessage.ReconnectIns,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending DisconnectRes message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build DisconnectRes message
    disconnect_res = ClientMessage.DisconnectRes(reason=reason)
    return ClientMessage(disconnect_res=disconnect_res), sleep_duration


def _get_properties(
    client: Client, get_properties_msg: ServerMessage.GetPropertiesIns
) -> ClientMessage:
    # Deserialize `get_properties` instruction
    get_properties_ins = serde.get_properties_ins_from_proto(get_properties_msg)

    # Request properties
    get_properties_res = maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )

    # Serialize response
    get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
    return ClientMessage(get_properties_res=get_properties_res_proto)


def _get_parameters(
    client: Client, get_parameters_msg: ServerMessage.GetParametersIns
) -> ClientMessage:
    # Deserialize `get_parameters` instruction
    get_parameters_ins = serde.get_parameters_ins_from_proto(get_parameters_msg)

    # Request parameters
    get_parameters_res = maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )

    # Serialize response
    get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
    return ClientMessage(get_parameters_res=get_parameters_res_proto)


def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)

    # Perform fit
    fit_res = maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )

    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)

    # Perform evaluation
    evaluate_res = maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )

    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)


def _example_response(client: Client, msg: ServerMessage.ExampleIns) -> ClientMessage:
    question, l = serde.example_msg_from_proto(msg)
    response, answer = client.example_response(question, l)
    example_res = serde.example_res_to_proto(response,answer)
    return ClientMessage(example_res=example_res)

# Step 1) Server sends shared vector_a to clients and they all send back vector_b
def _distribute_vector_a(client: Client, msg: ServerMessage.SendVectorAIns) -> ClientMessage:
    vector_a = serde.shared_vec_a_from_proto(msg)
    pubkey = client.generate_pubkey(vector_a)
    client_pub = serde.pub_key_b_to_proto(pubkey)
    return ClientMessage(send_vector_b_res=client_pub)

# Step 2) Server sends aggregated publickey allpub to clients and receive boolean confirmation
def _distribute_aggregated_pubkey(client: Client, msg: ServerMessage.SendAllpubIns) -> ClientMessage:
    aggregated_pubkey = serde.aggregated_pubkey_from_proto(msg)
    confirm = client.store_aggregated_pubkey(aggregated_pubkey)
    serde_res = serde.pubkey_confirmation_to_proto(confirm)
    return ClientMessage(send_allpub_res=serde_res)

# Step 3) After round, encrypt flat list of parameters into two lists (c0, c1)
def _request_encrypted_parameters(client: Client, msg: ServerMessage.RequestEncryptedIns) -> ClientMessage:
    request = serde.request_encrypted_from_proto(msg)
    c0, c1 = client.encrypt_parameters(request)
    serde_res = serde.send_encrypted_to_proto(c0, c1)
    return ClientMessage(send_encrypted_res=serde_res)

# Step 4) Send c1sum to clients and send back decryption share
def _request_decryption_shares(client: Client, msg: ServerMessage.SendCsumIns) -> ClientMessage:
    request = serde.send_csum1_from_proto(msg)
    d = client.compute_decryption_share(request)
    serde_res = serde.send_decryption_share_to_proto(d)
    return ClientMessage(send_dec_share_res=serde_res)

# Step 5) Send updated model weights to clients and return confirmation
def _request_modelupdate_confirmation(client: Client, msg: ServerMessage.SendNewWeightsIns) -> ClientMessage:
    request = serde.send_new_weights_from_proto(msg)
    confirm = client.receive_updated_weights(request)
    serde_res = serde.send_update_confirmation_to_proto(confirm)
    return ClientMessage(send_new_weights_res=serde_res)