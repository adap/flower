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


import os
from threading import Lock
from typing import Optional
import secrets
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
        # Store run ID to token mapping
        self.token_store: dict[int, bytes] = {}
        self.lock_token_store = Lock()

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
        print("DEBUG:\n")
        print(f"DEBUG: run_id={run_id}, is_reply={is_reply}, limit={limit}")
        print(f"DEBUG: Current message store: {self.msg_store}")

        # Normalize run_id to a list for consistent processing
        if isinstance(run_id, int):
            run_id = [run_id]

        with self.lock_msg_store:
            # Iterate through all messages in the store
            for object_id in list(self.msg_store.keys()):
                message = self.msg_store[object_id]

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

                # Remove the message from the store
                del self.msg_store[object_id]

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

    def get_run_ids_with_pending_messages(self) -> set[int]:
        """Get all stored run IDs with pending messages."""
        # Collect run IDs from messages
        with self.lock_msg_store:
            ret = {message.metadata.run_id for message in self.msg_store.values() if message.metadata.reply_to_message_id == ""}
        # Remove run IDs that have tokens stored (indicating they are in progress)
        with self.lock_token_store:
            ret -= set(self.token_store.keys())
            return ret

    def store_context(self, context: Context) -> None:
        """Store a context."""
        with self.lock_ctx_store:
            self.ctx_store[context.run_id] = context

    def get_context(self, run_id: int) -> Optional[Context]:
        """Retrieve a context by its run ID."""
        with self.lock_ctx_store:
            return self.ctx_store.get(run_id)

    def create_token(self, run_id: int) -> str:
        """Create a token for the given run ID."""
        token = secrets.token_hex(8)  # Generate a random token
        with self.lock_token_store:
            if run_id in self.token_store:
                raise ValueError("Token already created for this run ID")
            self.token_store[run_id] = token
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for a given run ID."""
        with self.lock_token_store:
            return self.token_store.get(run_id) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for a given run ID."""
        with self.lock_token_store:
            self.token_store.pop(run_id, None)

    def get_run_id_from_token(self, token: str) -> Optional[int]:
        """Get the run ID associated with a given token."""
        with self.lock_token_store:
            for run_id, stored_token in self.token_store.items():
                if stored_token == token:
                    return run_id
        return None
