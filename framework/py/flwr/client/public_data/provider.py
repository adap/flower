from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

class PublicDataProvider:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def get_samples(self, sample_ids: List[int]):
        xs = []
        for i in sample_ids:
            x, _ = self.dataset[i]   # ignore label (public unlabeled use)
            if isinstance(x, np.ndarray):
                x = torch.tensor(x)
            xs.append(x.unsqueeze(0))
        return torch.cat(xs, dim=0)  # [N, C, H, W]
