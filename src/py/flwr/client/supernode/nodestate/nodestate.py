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
"""Abstract base class NodeState."""

import abc
from typing import Optional

from ...clientapp.clientappio_servicer import ClientAppInputs, ClientAppOutputs


class NodeState(abc.ABC):
    """Abstract NodeState."""

    @abc.abstractmethod
    def set_clientapp_inputs(  # pylint: disable=R0913
        self, token: int, clientapp_inputs: ClientAppInputs
    ) -> None:
        """Set the clientapp inputs for a specified `run_id`.

        Parameters
        ----------
        token : int
            The unique token Run to be associated with the specified `ClientAppInputs`
        clientapp_inputs : ClientAppInputs
            The clientapp inputs to be associated with the specified `run_id`
        """

    @abc.abstractmethod
    def get_clientapp_inputs(self, token: int) -> Optional[ClientAppInputs]:
        """."""

    @abc.abstractmethod
    def set_clientapp_outputs(
        self, run_id: int, clientapp_outputs: ClientAppOutputs
    ) -> None:
        """."""

    @abc.abstractmethod
    def get_clientapp_outputs(self, clientapp_outputs: ClientAppOutputs) -> None:
        """."""
