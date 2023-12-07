# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


from typing import Optional, Tuple

from flwr.client.client import (
    Client,
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.message_handler.task_handler import (
    get_server_message_from_task_ins,
    wrap_client_message_in_task_res,
)
from flwr.client.secure_aggregation import SecureAggregationHandler
from flwr.client.typing import ClientFn
from flwr.client.workload_state import WorkloadState
from flwr.common import serde
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage


class UnexpectedServerMessage(Exception):
    """Exception indicating that the received message is unexpected."""


class UnknownServerMessage(Exception):
    """Exception indicating that the received message is unknown."""


def handle_control_message(task_ins: TaskIns) -> Tuple[Optional[TaskRes], int]:
    """Handle control part of the incoming message.

    Parameters
    ----------
    task_ins : TaskIns
        The task instruction coming from the server, to be processed by the client.

    Returns
    -------
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the client should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    server_msg = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)

    # SecAgg message
    if server_msg is None:
        return None, 0

    # ReconnectIns message
    field = server_msg.WhichOneof("msg")
    if field == "reconnect_ins":
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
        task_res = wrap_client_message_in_task_res(disconnect_msg)
        return task_res, sleep_duration

    # Any other message
    return None, 0


def handle(
    client_fn: ClientFn, state: WorkloadState, task_ins: TaskIns
) -> Tuple[TaskRes, WorkloadState]:
    """Handle incoming TaskIns from the server.

    Parameters
    ----------
    client_fn : ClientFn
        A callable that instantiates a Client.
    state : WorkloadState
        A dataclass storing the state for the workload being executed by the client.
    task_ins: TaskIns
        The task instruction coming from the server, to be processed by the client.

    Returns
    -------
    task_res : TaskRes
        The task response that should be returned to the server.
    """
    server_msg = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)
    if server_msg is None:
        # Instantiate the client
        client = client_fn("-1")
        client.set_state(state)
        # Secure Aggregation
        if task_ins.task.HasField("sa") and isinstance(
            client, SecureAggregationHandler
        ):
            # pylint: disable-next=invalid-name
            named_values = serde.named_values_from_proto(task_ins.task.sa.named_values)
            res = client.handle_secure_aggregation(named_values)
            task_res = TaskRes(
                task_id="",
                group_id="",
                workload_id=0,
                task=Task(
                    ancestry=[],
                    sa=SecureAggregation(named_values=serde.named_values_to_proto(res)),
                ),
            )
            return task_res, client.get_state()
        raise NotImplementedError()
    client_msg, updated_state = handle_legacy_message(client_fn, state, server_msg)
    task_res = wrap_client_message_in_task_res(client_msg)
    return task_res, updated_state


def handle_legacy_message(
    client_fn: ClientFn, state: WorkloadState, server_msg: ServerMessage
) -> Tuple[ClientMessage, WorkloadState]:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client_fn : ClientFn
        A callable that instantiates a Client.
    state : WorkloadState
        A dataclass storing the state for the workload being executed by the client.
    server_msg: ServerMessage
        The message coming from the server, to be processed by the client.

    Returns
    -------
    client_msg : ClientMessage
        The result message that should be returned to the server.
    """
    field = server_msg.WhichOneof("msg")

    # Must be handled elsewhere
    if field == "reconnect_ins":
        raise UnexpectedServerMessage()

    # Instantiate the client
    client = client_fn("-1")
    client.set_state(state)
    # Execute task
    message = None
    if field == "get_properties_ins":
        message = _get_properties(client, server_msg.get_properties_ins)
    if field == "get_parameters_ins":
        message = _get_parameters(client, server_msg.get_parameters_ins)
    if field == "fit_ins":
        message = _fit(client, server_msg.fit_ins)
    if field == "evaluate_ins":
        message = _evaluate(client, server_msg.evaluate_ins)
    if message:
        return message, client.get_state()
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
