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
"""Message Handler for Secure Aggregation (abstract base class)."""


from abc import ABC, abstractmethod

from flwr.common.typing import SecureAggregation


class SecureAggregationHandler(ABC):
    """Abstract base class for secure aggregation message handlers."""

    @abstractmethod
    # pylint: disable-next=invalid-name
    def handle_secure_aggregation(self, sa: SecureAggregation) -> SecureAggregation:
        """Handle incoming Secure Aggregation message and return results.

        Parameters
        ----------
        sa : SecureAggregation
            The SecureAggregation sub-message in Task message received
            from the server containing a dictionary of named values.

        Returns
        -------
        SecureAggregation
            The final/intermediate results of the Secure Aggregation protocol.
        """
