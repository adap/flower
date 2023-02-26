# Copyright 2023 Adap GmbH. All Rights Reserved.
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
"""Task handling."""


from typing import Optional, Tuple

from flwr.proto.fleet_pb2 import PullTaskInsResponse
from flwr.proto.task_pb2 import TaskIns
from flwr.proto.transport_pb2 import ServerMessage


def get_server_message(
    pull_task_ins_response: PullTaskInsResponse,
) -> Optional[Tuple[TaskIns, ServerMessage]]:
    """Get the first ServerMessage, if available."""

    # Extract a single ServerMessage from the response, if possible
    if len(pull_task_ins_response.task_ins_list) == 0:
        return None

    # Only evaluate the first message
    task_ins: TaskIns = pull_task_ins_response.task_ins_list[0]

    # Discard the message if it is not in
    # {GetPropertiesIns, GetParametersIns, FitIns, EvaluateIns}
    if (
        not task_ins.HasField("task")
        or not task_ins.task.HasField("legacy_server_message")
        or task_ins.task.legacy_server_message.WhichOneof("msg") == "reconnect_ins"
    ):
        return None

    return task_ins, task_ins.task.legacy_server_message
