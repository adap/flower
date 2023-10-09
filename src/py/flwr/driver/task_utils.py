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
"""Functions wrapping contents in Task."""


from typing import Dict, Union

from flwr.common import serde
from flwr.common.typing import (
    EvaluateIns,
    FitIns,
    GetParametersIns,
    GetPropertiesIns,
    ServerMessage,
    Value,
)
from flwr.proto import task_pb2, transport_pb2


def wrap_server_message_in_task(
    message: Union[
        ServerMessage, GetPropertiesIns, GetParametersIns, FitIns, EvaluateIns
    ]
) -> task_pb2.Task:
    """Wrap any server message/instruction in Task."""
    if isinstance(message, ServerMessage):
        server_message_proto = serde.server_message_to_proto(message)
    elif isinstance(message, GetPropertiesIns):
        server_message_proto = transport_pb2.ServerMessage(
            get_properties_ins=serde.get_properties_ins_to_proto(message)
        )
    elif isinstance(message, GetParametersIns):
        server_message_proto = transport_pb2.ServerMessage(
            get_parameters_ins=serde.get_parameters_ins_to_proto(message)
        )
    elif isinstance(message, FitIns):
        server_message_proto = transport_pb2.ServerMessage(
            fit_ins=serde.fit_ins_to_proto(message)
        )
    elif isinstance(message, EvaluateIns):
        server_message_proto = transport_pb2.ServerMessage(
            evaluate_ins=serde.evaluate_ins_to_proto(message)
        )
    return task_pb2.Task(legacy_server_message=server_message_proto)


def wrap_named_values_in_task(named_values: Dict[str, Value]) -> task_pb2.Task:
    """Wrap the `named_values` dictionary in SecureAggregation in Task."""
    return task_pb2.Task(
        sa=task_pb2.SecureAggregation(
            named_values=serde.named_values_to_proto(named_values)
        )
    )
