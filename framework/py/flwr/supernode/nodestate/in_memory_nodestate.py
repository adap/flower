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


import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

from flwr.common import Context, Message
from flwr.common.constant import FLWR_APP_TOKEN_LENGTH
from flwr.common.typing import Run

from .nodestate import NodeState


@dataclass
class MessageEntry:
    """Data class to represent a message entry."""

    message: Message
    is_retrieved: bool = False


class InMemoryNodeState(NodeState):  # pylint: disable=too-many-instance-attributes
    """In-memory NodeState implementation."""

    def __init__(self) -> None:
        # Store node_id
        self.node_id: int | None = None
        # Store Object ID to MessageEntry mapping
        self.msg_store: dict[str, MessageEntry] = {}
        self.lock_msg_store = Lock()
        # Store run ID to Run mapping
        self.run_store: dict[int, Run] = {}
        self.lock_run_store = Lock()
        # Store run ID to Context mapping
        self.ctx_store: dict[int, Context] = {}
        self.lock_ctx_store = Lock()
        # Store run ID to token mapping and token to run ID mapping
        self.token_store: dict[int, str] = {}
        self.token_to_run_id: dict[str, int] = {}
        self.lock_token_store = Lock()

    def set_node_id(self, node_id: int | None) -> None:
        """Set the node ID."""
        self.node_id = node_id

    def get_node_id(self) -> int:
        """Get the node ID."""
        if self.node_id is None:
            raise ValueError("Node ID not set")
        return self.node_id

    def store_message(self, message: Message) -> str | None:
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
        run_ids: Sequence[int] | None = None,
        is_reply: bool | None = None,
        limit: int | None = None,
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
        message_ids: Sequence[str] | None = None,
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

    def get_run(self, run_id: int) -> Run | None:
        """Retrieve a run by its ID."""
        with self.lock_run_store:
            return self.run_store.get(run_id)

    def store_context(self, context: Context) -> None:
        """Store a context."""
        with self.lock_ctx_store:
            self.ctx_store[context.run_id] = context

    def get_context(self, run_id: int) -> Context | None:
        """Retrieve a context by its run ID."""
        with self.lock_ctx_store:
            return self.ctx_store.get(run_id)

    def get_run_ids_with_pending_messages(self) -> Sequence[int]:
        """Retrieve run IDs that have at least one pending message."""
        # Collect run IDs from messages
        with self.lock_msg_store:
            ret = {
                entry.message.metadata.run_id
                for entry in self.msg_store.values()
                if entry.message.metadata.reply_to_message_id == ""
                and not entry.is_retrieved
            }

        # Remove run IDs that have tokens stored (indicating they are in progress)
        with self.lock_token_store:
            ret -= set(self.token_store.keys())
            return list(ret)

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        with self.lock_token_store:
            if run_id in self.token_store:
                return None  # Token already created for this run ID
            self.token_store[run_id] = token
            self.token_to_run_id[token] = run_id
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        with self.lock_token_store:
            return self.token_store.get(run_id) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        with self.lock_token_store:
            token = self.token_store.pop(run_id, None)
            if token is not None:
                self.token_to_run_id.pop(token, None)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        with self.lock_token_store:
            return self.token_to_run_id.get(token)
