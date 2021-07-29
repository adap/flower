# Copyright 2021 Adap GmbH. All Rights Reserved.
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
"""Shakespeare network model used in LEAF baselines."""

import torch
import torch.nn as nn

from flwr_baselines.dataloaders.leaf.shakespeare import LEAF_CHARACTERS


class ShakespeareLeafNet(nn.Module):  # type: ignore
    """Create Shakespeare model for LEAF baselines.

    Args:
        chars (str, optional): String of possible characters (letters+digits).
            Defaults to LEAF_CHARACTERS.
        seq_len (int, optional): Length of each sequence. Defaults to 80.
        hidden_size (int, optional): Size of hidden layer. Defaults to 256.
        embedding_dim (int, optional): Dimension of embedding. Defaults to 8.
    """

    def __init__(
        self,
        chars: str = LEAF_CHARACTERS,
        seq_len: int = 80,
        hidden_size: int = 256,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.dict_size = len(chars)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,  # Notice batch is first dim now
        )
        self.decoder = nn.Linear(self.hidden_size, self.dict_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forwards sentence to obtain next character.

        Args:
            sentence (torch.Tensor): Tensor containing indices of characters

        Returns:
            torch.Tensor: Vector encoding position of predicted character
        """
        encoded_seq = self.encoder(sentence)  # (batch, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(encoded_seq)  # (batch, seq_len, hidden_size)
        pred = self.decoder(h_n[-1])
        return pred


def get_model() -> nn.Module:
    """Returns the LEAF Shakespeare network.

    Returns:
        nn.Module: Implementation of LEAF Shakespeare network
    """
    return ShakespeareLeafNet()
