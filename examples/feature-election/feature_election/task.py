"""feature-election: Federated feature selection with Flower.

Task Module - Data Loading for Feature Election

Provides utilities for loading and partitioning tabular datasets
for federated feature selection.
"""

import logging
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 30,
    n_repeated: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create a synthetic high-dimensional dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_repeated: Number of repeated features
        random_state: Random seed

    Returns:
        Tuple of (DataFrame with features and target, feature names)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        random_state=random_state,
    )

    feature_names = [f"feature_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    logger.info(
        f"Created synthetic dataset: {n_samples} samples, {n_features} features"
    )
    return df, feature_names


def load_client_data(
    client_id: int,
    num_clients: int,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load data for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        split_strategy: Data splitting strategy ('stratified', 'random', 'non-iid')
        test_size: Fraction of data for validation
        random_state: Random seed
        dataset_params: Parameters for synthetic dataset generation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    # Create or load dataset
    if dataset_params is None:
        dataset_params = {
            "n_samples": 1000 * num_clients,  # Generate enough data for all clients
            "n_features": 100,
            "n_informative": 20,
            "n_redundant": 30,
            "n_repeated": 10,
        }

    # FIX: Use a fixed random_state for dataset generation so all clients
    # share the same underlying feature structure (informative/redundant features).
    # Only the data partitioning should be client-specific.
    df, feature_names = create_synthetic_dataset(
        random_state=random_state, **dataset_params
    )

    # Split data among clients
    client_df = split_data_for_client(
        df=df,
        client_id=client_id,
        num_clients=num_clients,
        strategy=split_strategy,
        random_state=random_state,
    )

    # Separate features and target
    X = client_df.drop(columns=["target"]).values
    y = client_df["target"].values

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state + client_id, stratify=y
    )

    logger.info(f"Client {client_id}: {len(X_train)} train, {len(X_val)} val samples")

    return X_train, y_train, X_val, y_val, feature_names


def split_data_for_client(
    df: pd.DataFrame,
    client_id: int,
    num_clients: int,
    strategy: str = "stratified",
    random_state: int = 42,
) -> pd.DataFrame:
    """Split dataset for a specific client based on strategy.

    Args:
        df: Full dataset
        client_id: Client identifier
        num_clients: Total number of clients
        strategy: Splitting strategy
        random_state: Random seed

    Returns:
        DataFrame for this client
    """
    if strategy == "stratified":
        return _split_stratified(df, client_id, num_clients, random_state)
    elif strategy == "random":
        return _split_random(df, client_id, num_clients, random_state)
    elif strategy == "non-iid":
        return _split_non_iid(df, client_id, num_clients, random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _split_stratified(
    df: pd.DataFrame, client_id: int, num_clients: int, random_state: int
) -> pd.DataFrame:
    """Stratified split maintaining class distribution."""
    # Simple approach: shuffle and partition
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Partition by client_id
    # Ensure equal-sized partitions
    rows_per_client = len(df) // num_clients
    start = client_id * rows_per_client
    end = start + rows_per_client

    # Handle last client getting remaining rows
    if client_id == num_clients - 1:
        end = len(df)

    return df_shuffled.iloc[start:end]


def _split_random(
    df: pd.DataFrame, client_id: int, num_clients: int, random_state: int
) -> pd.DataFrame:
    """Random split without stratification."""
    np.random.seed(random_state)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # Partition
    samples_per_client = len(df) // num_clients
    start = client_id * samples_per_client
    end = start + samples_per_client if client_id < num_clients - 1 else len(df)

    return df.iloc[indices[start:end]]


def _split_non_iid(
    df: pd.DataFrame,
    client_id: int,
    num_clients: int,
    random_state: int,
    classes_per_client: int = 2,
) -> pd.DataFrame:
    """Non-IID split where each client has data from limited classes.

    Args:
        df: Full dataset
        client_id: Client identifier
        num_clients: Total number of clients
        random_state: Random seed
        classes_per_client: Number of classes per client

    Returns:
        Non-IID data subset for client
    """
    target_col = "target"
    y = df[target_col].values

    # Encode labels if needed
    if y.dtype == object or y.dtype.name.startswith("str"):
        le = LabelEncoder()
        y = le.fit_transform(y)

    num_classes = len(np.unique(cast(np.ndarray, y)))

    # Assign classes to this client
    classes_for_client = []
    for i in range(classes_per_client):
        class_idx = (client_id * classes_per_client + i) % num_classes
        classes_for_client.append(class_idx)

    # Get indices for these classes
    client_indices: List[int] = []
    for class_idx in classes_for_client:
        class_indices = np.where(y == class_idx)[0]

        # Divide among clients that have this class
        clients_with_class = max(1, num_clients // (num_classes // classes_per_client))
        client_position = client_id % clients_with_class

        samples_per_client = len(class_indices) // clients_with_class
        if samples_per_client == 0:
            samples_per_client = len(class_indices)

        start = client_position * samples_per_client
        end = min(start + samples_per_client, len(class_indices))

        client_indices.extend(class_indices[start:end])

    return df.iloc[client_indices]


def prepare_federated_dataset(
    df: pd.DataFrame,
    target_col: str,
    num_clients: int,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare dataset for all clients in federated learning.

    Args:
        df: Full dataset
        target_col: Name of target column
        num_clients: Number of clients
        split_strategy: Data splitting strategy
        test_size: Fraction for validation
        random_state: Random seed

    Returns:
        List of (X_train, y_train, X_val, y_val) for each client
    """
    client_datasets = []

    for client_id in range(num_clients):
        # Split data for this client
        client_df = split_data_for_client(
            df, client_id, num_clients, split_strategy, random_state
        )

        # Separate features and target
        X = client_df.drop(columns=[target_col]).values
        y = client_df[target_col].values

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        client_datasets.append((X_train, y_train, X_val, y_val))

    logger.info(f"Prepared data for {num_clients} clients")
    return client_datasets


def load_custom_dataset(
    filepath: str,
    target_col: str,
    client_id: int,
    num_clients: int,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a custom dataset from file and prepare for a client.

    Args:
        filepath: Path to CSV file
        target_col: Name of target column
        client_id: Client identifier
        num_clients: Total number of clients
        split_strategy: Data splitting strategy
        test_size: Fraction for validation
        random_state: Random seed

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    # Load dataset
    df = pd.read_csv(filepath)

    # Get feature names
    feature_names = [col for col in df.columns if col != target_col]

    # Split for this client
    client_df = split_data_for_client(
        df, client_id, num_clients, split_strategy, random_state
    )

    X = client_df.drop(columns=[target_col]).values
    y = client_df[target_col].values
    if y.dtype == object or y.dtype.name.startswith("str"):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Loaded custom dataset for client {client_id}: "
        f"{len(X_train)} train, {len(X_val)} val samples"
    )

    return X_train, y_train, X_val, y_val, feature_names
