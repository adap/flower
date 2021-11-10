import torch
import pickle
import pandas as pd
from typing import Tuple
import smart_open
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    def __init__(self, data_path: str, transform=None) -> None:
        """Initialize IAIDataset
        Args:
            X (Union[pd.DataFrame, np.array]): DataFrame or numpy array of feature set
            y (Union[pd.DataFrame, np.array]): DataFrame or numpy array of labels
            X_type (torch.dtype, optional): Feature set type. Defaults to torch.float.
            y_type (torch.dtype, optional): Label type. Defaults to torch.long.
        """
        self.data_path = data_path
        self.transform = transform

        X, y = self._load_data(self.data_path)

        if len(X) != len(y):
            raise ValueError("Size of the feature dataset and labels do not match")

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        self.X, self.y = X, y

    def _load_data(self, data_path) -> Tuple:
        with smart_open.open(data_path, "rb") as f:
            output = pickle.load(f)
        return output["data"], output["labels"]

    def __len__(self) -> int:
        """Get the length of the dataset
        Returns:
            int: Length of the dataset
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a tuple of feature set and labels from the dataset
        given by the indices idx
        Args:
            idx (int): Dataset indices
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (feature_set, labels) tuple
        """

        if self.transform is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.X[idx], self.y[idx]
