# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Message."""


from dataclasses import dataclass
from typing import Optional

from .recordset import RecordSet


@dataclass
class Metadata:
    """A dataclass holding metadata associated with the current task.

    Parameters
    ----------
    run_id : int
        An identifier for the current run.
    task_id : str
        An identifier for the current task.
    group_id : str
        An identifier for grouping tasks. In some settings
        this is used as the FL round.
    ttl : str
        Time-to-live for this task.
    target_node_id : int
        The ID of the node that will receive the message.
    task_type : str
        A string that encodes the action to be executed on
        the receiving end.
    """

    run_id: int = 0
    task_id: str = ""
    group_id: str = ""
    ttl: str = ""
    target_node_id: Optional[int] = None
    task_type: str = ""


@dataclass
class Message:
    """State of your application from the viewpoint of the entity using it.

    Parameters
    ----------
    metadata : Metadata
        A dataclass including information about the task to be executed.
    message : RecordSet
        Holds records either sent by another entity (e.g. sent by the server-side
        logic to a client, or vice-versa) or that will be sent to it.
    """

    metadata: Metadata
    message: RecordSet

    def __init__(  # pylint: disable=too-many-arguments
        self,
        message: RecordSet,
        task_type: Optional[str] = None,
        target_node_id: Optional[int] = None,
        ttl: Optional[str] = None,
        metadata: Optional[Metadata] = None,
    ):
        self.message = message
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = Metadata(
                task_type=task_type if task_type is not None else "",
                target_node_id=target_node_id,
                ttl=ttl if ttl is not None else "",
            )
