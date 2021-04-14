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

        with open(path_to_pickle) as f:
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
