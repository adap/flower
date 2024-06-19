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
"""Flower in-memory Driver."""


import time
import warnings
from typing import Iterable, List, Optional, cast
from uuid import UUID

from flwr.common import DEFAULT_TTL, Message, Metadata, RecordSet
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.common.typing import Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.server.superlink.state import StateFactory

from .driver import Driver


class InMemoryDriver(Driver):
    """`InMemoryDriver` class provides an interface to the Driver API.

    Parameters
    ----------
    state_factory : StateFactory
        A StateFactory embedding a state that this driver can interface with.
    fab_id : str (default: None)
        The identifier of the FAB used in the run.
    fab_version : str (default: None)
        The version of the FAB used in the run.
    """

    def __init__(
        self,
        state_factory: StateFactory,
        fab_id: Optional[str] = None,
        fab_version: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> None:
        self._run_id = run_id
        self._fab_id = fab_id
        self._fab_ver = fab_version
        self.node = Node(node_id=0, anonymous=True)
        self.state = state_factory.state()

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.run_id == self.run.run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def _init_run(self) -> None:
        """Initialize the run."""
        # Run ID is not provided
        if self._run_id is None:
            self._fab_id = "" if self._fab_id is None else self._fab_id
            self._fab_ver = "" if self._fab_ver is None else self._fab_ver
            self._run_id = self.state.create_run(
                fab_id=self._fab_id, fab_version=self._fab_ver
            )
        # Run ID is provided
        elif self._fab_id is None or self._fab_ver is None:
            run = self.state.get_run(self._run_id)
            if run is None:
                raise RuntimeError(f"Cannot find the run with ID: {self._run_id}")
            self._fab_id = run.fab_id
            self._fab_ver = run.fab_version

    @property
    def run(self) -> Run:
        """Run ID."""
        self._init_run()
        return Run(
            run_id=cast(int, self._run_id),
            fab_id=cast(str, self._fab_id),
            fab_version=cast(str, self._fab_ver),
        )

    def create_message(  # pylint: disable=too-many-arguments
        self,
        content: RecordSet,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: Optional[float] = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )
        ttl_ = DEFAULT_TTL if ttl is None else ttl

        metadata = Metadata(
            run_id=self.run.run_id,
            message_id="",  # Will be set by the server
            src_node_id=self.node.node_id,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        return Message(metadata=metadata, content=content)

    def get_node_ids(self) -> List[int]:
        """Get node IDs."""
        return list(self.state.get_nodes(self.run.run_id))

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        task_ids: List[str] = []
        for msg in messages:
            # Check message
            self._check_message(msg)
            # Convert Message to TaskIns
            taskins = message_to_taskins(msg)
            # Store in state
            taskins.task.pushed_at = time.time()
            task_id = self.state.store_task_ins(taskins)
            if task_id:
                task_ids.append(str(task_id))

        return task_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        msg_ids = {UUID(msg_id) for msg_id in message_ids}
        # Pull TaskRes
        task_res_list = self.state.get_task_res(task_ids=msg_ids, limit=len(msg_ids))
        # Delete tasks in state
        self.state.delete_tasks(msg_ids)
        # Convert TaskRes to Message
        msgs = [message_from_taskres(taskres) for taskres in task_res_list]
        return msgs

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: Optional[float] = None,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull the reply messages.

        This method sends a list of messages to their destination node IDs and then
        waits for the replies. It continues to pull replies until either all replies are
        received or the specified timeout duration is exceeded.
        """
        # Push messages
        msg_ids = set(self.push_messages(messages))

        # Pull messages
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: List[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message for msg in res_msgs}
            )
            if len(msg_ids) == 0:
                break
            # Sleep
            time.sleep(3)
        return ret
