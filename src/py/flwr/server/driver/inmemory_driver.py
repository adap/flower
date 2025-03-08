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
from collections.abc import Iterable
from typing import Optional, cast
from uuid import UUID

from flwr.common import DEFAULT_TTL, Message, Metadata, RecordSet
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.typing import Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkStateFactory

from .driver import Driver


class InMemoryDriver(Driver):
    """`InMemoryDriver` class provides an interface to the ServerAppIo API.

    Parameters
    ----------
    state_factory : StateFactory
        A StateFactory embedding a state that this driver can interface with.
    pull_interval : float (default=0.1)
        Sleep duration between calls to `pull_messages`.
    """

    def __init__(
        self,
        state_factory: LinkStateFactory,
        pull_interval: float = 0.1,
    ) -> None:
        self._run: Optional[Run] = None
        self.state = state_factory.state()
        self.pull_interval = pull_interval
        self.node = Node(node_id=SUPERLINK_NODE_ID)

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.run_id == cast(Run, self._run).run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
            and message.metadata.delivered_at == ""
        ):
            raise ValueError(f"Invalid message: {message}")

    def set_run(self, run_id: int) -> None:
        """Initialize the run."""
        run = self.state.get_run(run_id)
        if run is None:
            raise RuntimeError(f"Cannot find the run with ID: {run_id}")
        self._run = run

    @property
    def run(self) -> Run:
        """Run ID."""
        return Run(**vars(cast(Run, self._run)))

    def create_message(  # pylint: disable=too-many-arguments,R0917
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
            run_id=cast(Run, self._run).run_id,
            message_id="",  # Will be set by the server
            src_node_id=self.node.node_id,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        return Message(metadata=metadata, content=content)

    def get_node_ids(self) -> Iterable[int]:
        """Get node IDs."""
        return self.state.get_nodes(cast(Run, self._run).run_id)

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        msg_ids: list[str] = []
        for msg in messages:
            # Check message
            self._check_message(msg)
            # Store in state
            msg_id = self.state.store_message_ins(msg)
            if msg_id:
                msg_ids.append(str(msg_id))

        return msg_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        msg_ids = {UUID(msg_id) for msg_id in message_ids}
        # Pull Messages
        message_res_list = self.state.get_message_res(message_ids=msg_ids)
        # Get IDs of Messages these replies are for
        message_ins_ids_to_delete = {
            UUID(msg_res.metadata.reply_to_message) for msg_res in message_res_list
        }
        # Delete
        self.state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

        return message_res_list

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
        ret: list[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message for msg in res_msgs}
            )
            if len(msg_ids) == 0:
                break
            # Sleep
            time.sleep(self.pull_interval)
        return ret
