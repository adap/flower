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
"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes


class Client(ABC):
    """Abstract base class for Flower clients."""

    @abstractmethod
    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters.

        Returns
        -------
        ParametersRes
            The current local model parameters.
        """

    @abstractmethod
    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    @abstractmethod
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """
