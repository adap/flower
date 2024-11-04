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

from ...clientapp.clientappio_servicer import ClientAppInputs


class NodeState(abc.ABC):
    """Abstract NodeState."""

    @abc.abstractmethod
    def set_clientapp_inputs(  # pylint: disable=R0913
        self, token: int, clientapp_inputs: ClientAppInputs
    ) -> None:
        """Set the ClientApp inputs for a specified `token`.

        Parameters
        ----------
        token : int
            The identifier of the ClientApp run for which to set the inputs.
        clientapp_inputs : ClientAppInputs
            The inputs for the ClientApp to be associated with the specified `token`.
        """

    @abc.abstractmethod
    def get_clientapp_inputs(self, token: int) -> Optional[ClientAppInputs]:
        """Get the ClientApp inputs for a specified `token`."""

    # @abc.abstractmethod
    # def set_clientapp_outputs(
    #     self, token: int, clientapp_outputs: ClientAppOutputs
    # ) -> None:
    #     """Set the ClientApp outputs for a specified `token`.

    #     Parameters
    #     ----------
    #     token : int
    #         The identifier of the ClientApp run for which to set the outputs.
    #     clientapp_inputs : ClientAppInputs
    #         The outputs from the ClientApp to be associated with the specified `token`.
    #     """

    # @abc.abstractmethod
    # def get_clientapp_outputs(self, token: int) -> Optional[ClientAppOutputs]:
    #     """Get the ClientApp outputs for a specified `token`."""
