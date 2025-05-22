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


from threading import Lock
from typing import Optional

from flwr.common import Context, Message
from flwr.common.typing import Run

from .nodestate import NodeState


class InMemoryNodeState(NodeState):
    """In-memory NodeState implementation."""

    def __init__(self) -> None:
        # Store node_id
        self.node_id: Optional[int] = None
        # Store Object ID to Message mapping
        self.msg_store: dict[str, Message] = {}
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

    def get_message(
        self,
        *,
        run_id: Optional[int | list[int]] = None,
        is_reply: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Message]:
        """Retrieve messages based on the specified filters."""
        selected_messages: dict[str, Message] = {}

        # Normalize run_id to a list for consistent processing
        if isinstance(run_id, int):
            run_id = [run_id]

        with self.lock_msg_store:
            # Iterate through all messages in the store
            for object_id, message in self.msg_store.items():
                # Skip messages whose run_id doesn't match the filter
                if run_id is not None:
                    if message.metadata.run_id not in run_id:
                        continue

                # If is_reply filter is set, filter for reply/non-reply messages
                if is_reply is not None:
                    is_reply_message = message.metadata.reply_to_message_id != ""
                    # XOR logic to filter mismatched types (reply vs non-reply)
                    if is_reply ^ is_reply_message:
                        continue

                # Add the message to the result set
                selected_messages[object_id] = message

                # Stop if the number of collected messages reaches the limit
                if limit is not None and len(selected_messages) >= limit:
                    break

        return selected_messages

    def store_message(self, message: Message, object_id: str) -> None:
        """Store a message."""
        with self.lock_msg_store:
            self.msg_store[object_id] = message

    def store_run(self, run: Run) -> None:
        """Store a run."""
        with self.lock_run_store:
            self.run_store[run.run_id] = run

    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve a run by its ID."""
        with self.lock_run_store:
            return self.run_store.get(run_id)

    def get_run_ids(self) -> list[int]:
        """Get all stored run IDs."""
        with self.lock_run_store:
            return list(self.run_store.keys())

    def store_context(self, context: Context, run_id: int) -> None:
        """Store a context."""
        with self.lock_ctx_store:
            self.ctx_store[run_id] = context

    def get_context(self, run_id: int) -> Optional[Context]:
        """Retrieve a context by its run ID."""
        with self.lock_ctx_store:
            return self.ctx_store.get(run_id)
