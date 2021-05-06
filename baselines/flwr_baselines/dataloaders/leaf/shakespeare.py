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

import json
import pickle
from os import PathLike

import numpy as np
from torch.utils.data import Dataset

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class ShakespeareDataset(Dataset):
    """Creates a PyTorch Dataset for Leaf Shakespeare.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle: PathLike):

        self.CHARACTERS = LEAF_CHARACTERS
        self.NUM_LETTERS = len(self.CHARACTERS)  # 80
        self.x, self.y = [], []

        with open(path_to_pickle, "rb") as f:
            data = pickle.load(f)
            self.x = data["x"]
            self.y = data["y"]
            self.idx = data["idx"]
            self.char = data["character"]

    def word_to_indices(self, word: str):
        indices = [self.CHARACTERS.find(c) for c in word]
        return indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = np.array(self.word_to_indices(self.x[idx]))
        y = np.array(self.CHARACTERS.find(self.y[idx]))
        return x, y
