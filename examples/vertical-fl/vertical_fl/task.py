from pathlib import Path
from logging import WARN
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn as nn
from flwr.common.logger import log

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

NUM_VERTICAL_SPLITS = 3


def _bin_age(age_series):
    bins = [-np.inf, 10, 40, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df):
    # Convert 'Age' to numeric, coercing errors to NaN
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    all_keywords = set(df.columns)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    return df, all_keywords


def process_dataset():

    df = pd.read_csv(Path(__file__).parents[1] / "data/train.csv")
    processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
    return _create_features(processed_df)


def load_data(partition_id: int, num_partitions: int):
    """Partition the data vertically and then horizontally.

    We create three sets of features representing three types of nodes participating in
    the federation.

    [{'Cabin', 'Parch', 'Pclass'}, {'Sex', 'Title'}, {'Age', 'Embarked', 'Fare',
    'SibSp', 'Survived'}]

    Once the whole dataset is split vertically and a set of features is selected based
    on mod(partition_id, 3), it is split horizontally into `ceil(num_partitions/3)`
    partitions. This function returns the partition with index `partition_id % 3`.
    """

    if num_partitions != NUM_VERTICAL_SPLITS:
        log(
            WARN,
            "To run this example with num_partitions other than 3, you need to update how "
            "the Vertical FL training is performed. This is because the shapes of the "
            "gradients migh not be the same along the first dimension.",
        )

    # Read whole dataset and process
    processed_df, features_set = process_dataset()

    # Vertical Split and select
    v_partitions = _partition_data_vertically(processed_df, features_set)
    v_split_id = np.mod(partition_id, NUM_VERTICAL_SPLITS)
    v_partition = v_partitions[v_split_id]

    # Comvert to HuggingFace dataset
    dataset = Dataset.from_pandas(v_partition)

    # Split horizontally with Flower Dataset partitioner
    num_h_partitions = int(np.ceil(num_partitions / NUM_VERTICAL_SPLITS))
    partitioner = IidPartitioner(num_partitions=num_h_partitions)
    partitioner.dataset = dataset

    # Extract partition of the `ClientApp` calling this function
    partition = partitioner.load_partition(partition_id % num_h_partitions)
    partition.remove_columns(["Survived"])

    return partition.to_pandas(), v_split_id


def _partition_data_vertically(df, all_keywords):
    partitions = []
    keywords_sets = [{"Parch", "Cabin", "Pclass"}, {"Sex", "Title"}]
    keywords_sets.append(all_keywords - keywords_sets[0] - keywords_sets[1])

    for keywords in keywords_sets:
        partitions.append(
            df[
                list(
                    {
                        col
                        for col in df.columns
                        for kw in keywords
                        if kw in col or "Survived" in col
                    }
                )
            ]
        )

    return partitions


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)
