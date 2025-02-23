import pandas as pd
from torch.utils.data import random_split
from .tolkien_dataset import TolkienDataset


class TolkienDatasetBuilder:
    def __init__(self, filename, model_name, val_percent=0.1):
        self.filename = filename
        self.val_percent = val_percent
        self.model_name = model_name

    def build_datasets(self):
        """
        Reads in the dataset
        Returns a tuple of training and validation datasets
        """
        print(f"Reading data from {self.filename}...")

        df = pd.read_json(self.filename)
        df = df.rename(columns={0: "sentences"})
        dataset = TolkienDataset(df, self.model_name)

        train_dataset, val_dataset = self.random_split_dataset(dataset)

        return train_dataset, val_dataset

    def random_split_dataset(self, df):
        """
        Takes a pandas dataframe and splits it into a training and validation set
        Returns a tuple of training and validation datasets
        """
        val_size = int(len(df) * self.val_percent)
        train_size = len(df) - val_size
        return random_split(df, [train_size, val_size])  # returns a tuple
