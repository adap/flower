# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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

from logging import WARN
from typing import Optional, Tuple, cast

from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.numpy_client import NumPyClient
from flwr.client.typing import ClientFnExt
from flwr.common import ConfigsRecord, Context, Message, Metadata, RecordSet, log
from flwr.common.constant import MessageType, MessageTypeLegacy
from flwr.common.recordset_compat import (
    evaluateres_to_recordset,
    fitres_to_recordset,
    getparametersres_to_recordset,
    getpropertiesres_to_recordset,
    recordset_to_evaluateins,
    recordset_to_fitins,
    recordset_to_getparametersins,
    recordset_to_getpropertiesins,
)
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    Reason,
    ServerMessage,
)


class UnexpectedServerMessage(Exception):
    """Exception indicating that the received message is unexpected."""


class UnknownServerMessage(Exception):
    """Exception indicating that the received message is unknown."""


def handle_control_message(message: Message) -> Tuple[Optional[Message], int]:
    """Handle control part of the incoming message.

    Parameters
    ----------
    message : Message
        The Message coming from the server, to be processed by the client.

    Returns
    -------
    message : Optional[Message]
        Message to be sent back to the server. If None, the client should
        continue to process messages from the server.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    """
    if message.metadata.message_type == "reconnect":
        # Retrieve ReconnectIns from recordset
        recordset = message.content
        seconds = cast(int, recordset.configs_records["config"]["seconds"])
        # Construct ReconnectIns and call _reconnect
        disconnect_msg, sleep_duration = _reconnect(
            ServerMessage.ReconnectIns(seconds=seconds)
        )
        # Store DisconnectRes in recordset
        reason = cast(int, disconnect_msg.disconnect_res.reason)
        recordset = RecordSet()
        recordset.configs_records["config"] = ConfigsRecord({"reason": reason})
        out_message = message.create_reply(recordset)
        # Return TaskRes and sleep duration
        return out_message, sleep_duration

    # Any other message
    return None, 0


def handle_legacy_message_from_msgtype(
    client_fn: ClientFnExt, message: Message, context: Context
) -> Message:
    """Handle legacy message in the inner most mod."""
    client = client_fn(context)

    # Check if NumPyClient is returend
    if isinstance(client, NumPyClient):
        client = client.to_client()
        log(
            WARN,
            "Deprecation Warning: The `client_fn` function must return an instance "
            "of `Client`, but an instance of `NumpyClient` was returned. "
            "Please use `NumPyClient.to_client()` method to convert it to `Client`.",
        )

    client.set_context(context)

    message_type = message.metadata.message_type

    # Handle GetPropertiesIns
    if message_type == MessageTypeLegacy.GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recordset_to_getpropertiesins(message.content),
        )
        out_recordset = getpropertiesres_to_recordset(get_properties_res)
    # Handle GetParametersIns
    elif message_type == MessageTypeLegacy.GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recordset_to_getparametersins(message.content),
        )
        out_recordset = getparametersres_to_recordset(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif message_type == MessageType.TRAIN:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recordset_to_fitins(message.content, keep_input=True),
        )
        out_recordset = fitres_to_recordset(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif message_type == MessageType.EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recordset_to_evaluateins(message.content, keep_input=True),
        )
        out_recordset = evaluateres_to_recordset(evaluate_res)
    else:
        raise ValueError(f"Invalid message type: {message_type}")

    # Return Message
    return message.create_reply(out_recordset)


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


def validate_out_message(out_message: Message, in_message_metadata: Metadata) -> bool:
    """Validate the out message."""
    out_meta = out_message.metadata
    in_meta = in_message_metadata
    if (  # pylint: disable-next=too-many-boolean-expressions
        out_meta.run_id == in_meta.run_id
        and out_meta.message_id == ""  # This will be generated by the server
        and out_meta.src_node_id == in_meta.dst_node_id
        and out_meta.dst_node_id == in_meta.src_node_id
        and out_meta.reply_to_message == in_meta.message_id
        and out_meta.group_id == in_meta.group_id
        and out_meta.message_type == in_meta.message_type
        and out_meta.created_at > in_meta.created_at
    ):
        return True
    return False
