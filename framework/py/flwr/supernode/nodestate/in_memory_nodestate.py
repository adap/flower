# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""In-memory NodeState implementation."""


from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from flwr.common import Context, Message
from flwr.common.typing import Run

from .nodestate import NodeState


@dataclass
class MessageEntry:
    """Data class to represent a message entry."""

    message: Message
    is_retrieved: bool = False


class InMemoryNodeState(NodeState):
    """In-memory NodeState implementation."""

    def __init__(self) -> None:
        # Store node_id
        self.node_id: Optional[int] = None
        # Store Object ID to MessageEntry mapping
        self.msg_store: dict[str, MessageEntry] = {}
        self.lock_msg_store = Lock()
        # Store run ID to Run mapping
        self.run_store: dict[int, Run] = {}
        self.lock_run_store = Lock()
        # Store run ID to Context mapping
        self.ctx_store: dict[int, Context] = {}
        self.lock_ctx_store = Lock()

    def set_node_id(self, node_id: Optional[int]) -> None:
        """Set the node ID."""
        self.node_id = node_id

    def get_node_id(self) -> int:
        """Get the node ID."""
        if self.node_id is None:
            raise ValueError("Node ID not set")
        return self.node_id

    def store_message(self, message: Message) -> Optional[str]:
        """Store a message."""
        with self.lock_msg_store:
            msg_id = message.metadata.message_id
            if msg_id == "" or msg_id in self.msg_store:
                return None
            self.msg_store[msg_id] = MessageEntry(message=message)
            return msg_id

    def get_messages(
        self,
        *,
        run_ids: Optional[Sequence[int]] = None,
        is_reply: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Message]:
        """Retrieve messages based on the specified filters."""
        selected_messages: list[Message] = []

        with self.lock_msg_store:
            # Iterate through all messages in the store
            for object_id in list(self.msg_store.keys()):
                entry = self.msg_store[object_id]
                message = entry.message

                # Skip messages that have already been retrieved
                if entry.is_retrieved:
                    continue

                # Skip messages whose run_id doesn't match the filter
                if run_ids is not None:
                    if message.metadata.run_id not in run_ids:
                        continue

                # If is_reply filter is set, filter for reply/non-reply messages
                if is_reply is not None:
                    is_reply_message = message.metadata.reply_to_message_id != ""
                    # XOR logic to filter mismatched types (reply vs non-reply)
                    if is_reply ^ is_reply_message:
                        continue

                # Add the message to the result set
                selected_messages.append(message)

                # Mark the message as retrieved
                entry.is_retrieved = True

                # Stop if the number of collected messages reaches the limit
                if limit is not None and len(selected_messages) >= limit:
                    break

        return selected_messages

    def delete_messages(
        self,
        *,
        message_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Delete messages based on the specified filters."""
        with self.lock_msg_store:
            if message_ids is None:
                # If no message IDs are provided, clear the entire store
                self.msg_store.clear()
                return

            # Remove specified messages from the store
            for msg_id in message_ids:
                self.msg_store.pop(msg_id, None)

    def store_run(self, run: Run) -> None:
        """Store a run."""
        with self.lock_run_store:
            self.run_store[run.run_id] = run

    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve a run by its ID."""
        with self.lock_run_store:
            return self.run_store.get(run_id)

    def store_context(self, context: Context) -> None:
        """Store a context."""
        with self.lock_ctx_store:
            self.ctx_store[context.run_id] = context

    def get_context(self, run_id: int) -> Optional[Context]:
        """Retrieve a context by its run ID."""
        with self.lock_ctx_store:
            return self.ctx_store.get(run_id)
