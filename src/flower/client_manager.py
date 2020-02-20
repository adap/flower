# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Flower ClientManager."""

from abc import ABC, abstractmethod

from .client import Client


class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def register(self, client: Client) -> bool:
        """Register Flower Client instance.

        Returns:
            bool: Indicating if registration was successful
        """
        raise NotImplementedError()

    @abstractmethod
    def unregister(self, client: Client) -> None:
        """Unregister Flower Client instance."""
        raise NotImplementedError()
