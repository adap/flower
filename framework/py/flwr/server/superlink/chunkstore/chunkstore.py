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

import abc
from typing import Optional
from uuid import UUID

from flwr.proto.chunk_pb2 import Chunk  # pylint: disable=E0611


class ChunkStore(abc.ABC):
    """Abstract ChunkStore."""

    @abc.abstractmethod
    def store_chunk(self, message_id: UUID, chunk: Chunk) -> None:
        """Store one chunk in the store."""

    @abc.abstractmethod
    def get_chunk(self, message_id: UUID) -> Optional[Chunk]:
        """Retrieve one chunk from the store."""

    @abc.abstractmethod
    def delete_chunks(self, message_id: UUID) -> None:
        """Delete all chunks associated to a Message."""
