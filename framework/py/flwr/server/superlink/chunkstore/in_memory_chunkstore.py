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
"""Flower ChunkStore."""

import threading
from typing import Optional
from uuid import UUID

from flwr.common.constant import NODE_ID_NUM_BYTES
from flwr.proto.chunk_pb2 import Chunk  # pylint: disable=E0611

from ..linkstate.utils import generate_rand_int_from_bytes


class InMemoryChunkStore:
    """In-memory ChunkStore implementation."""

    def __init__(self) -> None:
        """."""
        # Stores chunks: {chunk-id, Chunk}
        self.chunk_store: dict[UUID, Chunk] = {}
        # Keeps a mapping of which Chunks belong to which Message-Id
        # Can be used to erase Chunks when the Message is also deleted
        self.message_id_to_chunk_mapping: dict[UUID, set[UUID]] = {}

        # Keep track of chunks that have been retrieved already
        # so we don't return them twice (later it could be used to
        # erase them -- this is not considered in the RFC)
        self.retrieved: set[UUID] = set()

        # Need to consider lock
        self.lock = threading.RLock()

    def store_chunk(self, message_id: UUID, chunk: Chunk) -> None:
        """."""
        # generate UUID for the chunk
        chunk_uuid = generate_rand_int_from_bytes(
            num_bytes=NODE_ID_NUM_BYTES
        )  # same technique as with messages

        # add to mapping (or initialize if first chunk for message -- not shown)
        self.message_id_to_chunk_mapping[message_id].add(chunk_uuid)

        # store chunk
        self.chunk_store[chunk_uuid] = chunk

    def get_chunk(self, message_id: UUID) -> Optional[Chunk]:
        """."""
        # extract one chunk that belongs to specified message
        ids_chunks_not_fetched = (
            self.message_id_to_chunk_mapping[message_id] - self.retrieved
        )
        chunk_id_to_return = next(iter(ids_chunks_not_fetched))
        # add chunk UUID to the set of retrieved chunks so it's not pulled again
        self.retrieved.add(chunk_id_to_return)

        # return the chunk
        return self.chunk_store[chunk_id_to_return]
