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
from typing import Iterable, List, Optional
from uuid import UUID

from flwr.common import DEFAULT_TTL, Message, Metadata, RecordSet
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.server.superlink.state import InMemoryState, StateFactory

from .driver import Driver


class InMemoryDriver(Driver):
    """`InMemoryDriver` class provides an interface to the Driver API.

    Parameters
    ----------
    state_factory : StateFactory
        A StateFactory embedding a state designed for in-memory communication.
    """

    def __init__(
        self,
        state_factory: StateFactory,
    ) -> None:
        self.run_id: Optional[int] = None
        self.node = Node(node_id=0, anonymous=True)
        state = state_factory.state()

        if not isinstance(state, InMemoryState):
            raise NotImplementedError(
                f"{self.__class__.__name__} is only compatible with an `InMemoryState` "
                f"state. You used {type(state)}"
            )

        self.state = state

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.run_id == self.run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def _get_run_id(self) -> int:
        if self.run_id is None:
            self.run_id = self.state.create_run() # TODO: how?
        return self.run_id

    def create_message(
        self,
        content: RecordSet,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        run_id = self._get_run_id()
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )
        ttl_ = DEFAULT_TTL if ttl is None else ttl

        metadata = Metadata(
            run_id=run_id,
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
        run_id = self._get_run_id()
        return list(self.state.get_nodes(run_id))

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
            res = self.state.store_task_ins(taskins)
            if res:
                task_ids.append(str(res))

        return task_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        msg_ids = {UUID(msg_id) for msg_id in message_ids}
        # Pull TaskRes
        task_res_list = self.state.get_task_res(task_ids=msg_ids, limit=1)
        # Convert TaskRes to Message
        msgs = [message_from_taskres(taskres) for taskres in task_res_list]
        return msgs
