# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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


from typing import Optional

from flwr.proto.fleet_pb2 import PullTaskInsResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611


def validate_task_ins(task_ins: TaskIns) -> bool:
    """Validate a TaskIns before it entering the message handling process.

    Parameters
    ----------
    task_ins: TaskIns
        The task instruction coming from the server.

    Returns
    -------
    is_valid: bool
        True if the TaskIns is deemed valid and therefore suitable for
        undergoing the message handling process, False otherwise.
    """
    if not (task_ins.HasField("task") and task_ins.task.HasField("recordset")):
        return False
    return True


def get_task_ins(
    pull_task_ins_response: PullTaskInsResponse,
) -> Optional[TaskIns]:
    """Get the first TaskIns, if available."""
    # Extract a single ServerMessage from the response, if possible
    if len(pull_task_ins_response.task_ins_list) == 0:
        return None

    # Only evaluate the first message
    task_ins: TaskIns = pull_task_ins_response.task_ins_list[0]

    return task_ins
