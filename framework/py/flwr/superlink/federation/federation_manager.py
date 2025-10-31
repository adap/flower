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
"""Abstract base class FederationManager."""


from abc import ABC, abstractmethod

from flwr.app import Message


class FederationManager(ABC):

    @abstractmethod
    def exists(self, federation_name: str) -> bool:
        """Check if a federation exists."""

    @abstractmethod
    def is_member(self, federation_name: str, flwr_aid: str) -> bool:
        """Check if a member of the federation."""

        # ControlServicer calls this method when processing each request

        # This method should also be checked either periodically or
        # by ServerAppIoServicer to terminate a run if a user stops
        # being a member.

    @abstractmethod
    def filter_nodes(self, node_ids: list[int], federation_name: str) -> list[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""

        # ServerAppIo.GetNodes calls this method before returning to Grid

    @abstractmethod
    def filter_messages(
        self, messages: list[Message], fedeation_name: str
    ) -> list[Message]:
        """Given a list of messages that arrive to ServerAppIoServicer filter out
        messages that include destination nodes that aren't part of the federation."""

        # ServerAppIo.PushMessage calls this method before storing Messages
        # that the Grid are stored in the LinkState. This also prevents
        # the objects from being pre-registered if SuperNodes aren't in
        # the federation.
