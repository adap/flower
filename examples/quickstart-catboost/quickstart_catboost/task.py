"""quickstart-catboost: A Flower / CatBoost app."""

import json
import tempfile

from datasets import Dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


fds = None  # Cache FederatedDataset


def model_temp_file(model, dump):
    """Generate temp file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        if dump:
            json.dump(json.loads(model), open(tmp.name, "w+"))
        else:
            model.save_model(tmp.name, format="json")
        tmp_path = tmp.name
    return tmp_path


def preprocess(partition: Dataset):
    """Pre-process adult-census-income data."""
    # Divide data on each node: 80% train, 20% test
    # partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition.train_test_split(test_size=0.003, seed=42)

    # Load train and test data
    # train_ds = partition_train_test["train"]
    train_ds = partition_train_test["test"]
    test_ds = partition_train_test["test"]
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    # Construct labels
    train_df["income"] = train_df["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )
    test_df["income"] = test_df["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )
    feature_cols = [col for col in train_df.columns if col != "income"]
    X_train = train_df[feature_cols]
    y_train = train_df["income"]
    X_test = test_df[feature_cols]
    y_test = test_df["income"]

    return (X_train, y_train), (X_test, y_test)


def load_data(partition_id: int, num_partitions: int):
    """Load partition adult-census-income data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    train_data, test_data = preprocess(partition)
    cat_features = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]

    return train_data, test_data, cat_features
