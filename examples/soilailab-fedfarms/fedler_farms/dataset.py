from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner 


def make_federated_dataset(hf_dataset: str, partition_by: str = "farm") -> FederatedDataset:
    """Create a FederatedDataset partitioned by a natural-id column (e.g., 'farm')."""
    partitioner = NaturalIdPartitioner(partition_by=partition_by)
    return FederatedDataset(dataset=hf_dataset, partitioners={"train": partitioner})


def load_client_partition(
    fds: FederatedDataset,
    partition_id: int,
    feature_cols: List[str],
    target_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one client's partition from the partitioned train split."""
    part = fds.load_partition(partition_id, "train")
    df = part.to_pandas()

    missing = (set(feature_cols) | set(target_cols)) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train partition: {sorted(missing)}")

    X = df[feature_cols].reset_index(drop=True)
    y = df[target_cols].reset_index(drop=True)
    return X, y


def load_global_test_split(
    fds: FederatedDataset,
    feature_cols: List[str],
    target_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full centralized test split."""
    ds_test = fds.load_split("test")
    df = ds_test.to_pandas()

    missing = (set(feature_cols) | set(target_cols)) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test split: {sorted(missing)}")

    X = df[feature_cols].reset_index(drop=True)
    y = df[target_cols].reset_index(drop=True)
    return X, y