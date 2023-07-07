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
"""Flower client app."""


from abc import ABC
from typing import Dict, Tuple, List

from flwr.common import Config, NDArrays, Scalar


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Returns a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """


def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


def has_get_parameters(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_parameters."""
    return type(client).get_parameters != NumPyClient.get_parameters


def has_fit(client: NumPyClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit != NumPyClient.fit


def has_evaluate(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate."""
    return type(client).evaluate != NumPyClient.evaluate

def has_example_response(client: NumPyClient) -> bool:
    return callable(getattr(client, "example_response", None))

# Step 1) Server sends shared vector_a to clients and they all send back vector_b
def has_generate_pubkey(client: NumPyClient) -> bool:
    return callable(getattr(client, "generate_pubkey", None))

# Step 2) Server sends aggregated publickey allpub to clients and receive boolean confirmation
def has_store_aggregated_pubkey(client: NumPyClient) -> bool:
    return callable(getattr(client, "store_aggregated_pubkey", None))

# Step 3) After round, encrypt flat list of parameters into two lists (c0, c1)
def has_encrypt_parameters(client: NumPyClient) -> bool:
    return callable(getattr(client, "encrypt_parameters", None))

# Step 4) Send c1sum to clients and send back decryption share
def has_compute_decryption_share(client: NumPyClient) -> bool:
    return callable(getattr(client, "compute_decryption_share", None))

# Step 5) Send updated model weights to clients and return confirmation
def has_receive_updated_weights(client: NumPyClient) -> bool:
    return callable(getattr(client, "receive_updated_weights", None))

"""
def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
    response = "Here you go Alice!"
    answer = sum(l)
    return response, answer
"""