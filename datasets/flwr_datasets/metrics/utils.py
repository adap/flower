# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Utils for metrics computation."""


import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flwr_datasets.partitioner import Partitioner


def compute_counts(
    partitioner: Partitioner,
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the counts of unique values in a given column in the partitions.

    Take into account all possible labels in dataset when computing count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying label based on which the count will be calculated.
    verbose_names : bool
        Whether to use verbose versions of the values in the column specified by
        `column_name`. The verbose values are possible to extract if the column is a
        feature of type `ClassLabel`.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    dataframe: pd.DataFrame
        DataFrame where the row index represent the partition id and the column index
        represent the unique values found in column specified by `column_name`
        (e.g. represeting the labels). The value of the dataframe.loc[i, j] represents
        the count of the label j, in the partition of index i.

    Examples
    --------
    Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.metrics import compute_counts
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> counts_dataframe = compute_counts(
    >>>     partitioner=partitioner,
    >>>     column_name="label"
    >>> )
    """
    if column_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'column_name': '{column_name}' is not present in the "
            f"dataset. The dataset contains columns {partitioner.dataset.column_names}."
        )

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partition = partitioner.load_partition(0)

    try:
        # Unique labels are needed to represent the correct count of each class
        # (some of the classes can have zero samples that's why this
        # adjustment is needed)
        unique_labels = partition.features[column_name].str2int(
            partition.features[column_name].names
        )
    except AttributeError:  # If the column_name is not formally a Label
        unique_labels = partitioner.dataset.unique(column_name)

    partition_id_to_label_absolute_size = {}
    for partition_id in range(max_num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_id_to_label_absolute_size[partition_id] = _compute_counts(
            partition[column_name], unique_labels
        )

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_label_absolute_size, orient="index"
    )
    dataframe.index.name = "Partition ID"

    if verbose_names:
        # Adjust the column name values of the dataframe
        current_labels = dataframe.columns
        try:
            legend_names = partitioner.dataset.features[column_name].int2str(
                [int(v) for v in current_labels]
            )
            dataframe.columns = legend_names
        except AttributeError:
            warnings.warn(
                "The verbose names can not be established. "
                "The column specified by 'column_name' needs to be of type "
                "'ClassLabel' to create a verbose names. "
                "The available names will used.",
                stacklevel=1,
            )
    return dataframe


def compute_frequencies(
    partitioner: Partitioner,
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the frequencies of unique values in a given column in the partitions.

    The frequencies sum up to 1 for a given partition id. This function takes into
    account all possible labels in the dataset when computing the count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying label based on which the count will be calculated.
    verbose_names : bool
        Whether to use verbose versions of the values in the column specified by
        `column_name`. The verbose value are possible to extract if the column is a
        feature of type `ClassLabel`.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    dataframe: pd.DataFrame
        DataFrame where the row index represent the partition id and the column index
        represent the unique values found in column specified by `column_name`
        (e.g. represeting the labels). The value of the dataframe.loc[i, j] represnt
        the ratio of the label j to the total number of sample of in partition i.

    Examples
    --------
    Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.metrics import compute_frequencies
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> counts_dataframe = compute_frequencies(
    >>>     partitioner=partitioner,
    >>>     column_name="label"
    >>> )
    """
    dataframe = compute_counts(
        partitioner, column_name, verbose_names, max_num_partitions
    )
    dataframe = dataframe.div(dataframe.sum(axis=1), axis=0)
    return dataframe


def _compute_counts(
    labels: Union[List[int], List[str]], unique_labels: Union[List[int], List[str]]
) -> pd.Series:
    """Compute the count of labels when taking into account all possible labels.

    Also known as absolute frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
    label_counts: pd.Series
        The pd.Series with label as indices and counts as values.
    """
    if len(unique_labels) != len(set(unique_labels)):
        raise ValueError("unique_labels must contain unique elements only.")
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros


def _compute_frequencies(
    labels: Union[List[int], List[str]], unique_labels: Union[List[int], List[str]]
) -> pd.Series:
    """Compute the distribution of labels when taking into account all possible labels.

    Also known as relative frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
        The pd.Series with label as indices and probabilities as values.
    """
    counts = _compute_counts(labels, unique_labels)
    if len(labels) == 0:
        frequencies = counts.astype(float)
        return frequencies
    frequencies = counts.divide(len(labels))
    return frequencies


def compute_counts(
    labels: Union[List[int], List[str]], unique_labels: Union[List[int], List[str]]
) -> pd.Series:
    """Compute the count of labels when taking into account all possible labels.

    Also known as absolute frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
    label_counts: pd.Series
        The pd.Series with label as indices and counts as values.
    """
    if len(unique_labels) != len(set(unique_labels)):
        raise ValueError("unique_labels must contain unique elements only.")
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros


def compute_frequency(
    labels: Union[List[int], List[str]], unique_labels: Union[List[int], List[str]]
) -> pd.Series:
    """Compute the distribution of labels when taking into account all possible labels.

    Also known as relative frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
        The pd.Series with label as indices and probabilities as values.
    """
    counts = compute_counts(labels, unique_labels)
    if len(labels) == 0:
        counts = counts.astype(float)
        return counts
    counts = counts.divide(len(labels))
    return counts


def get_distros(
    targets_per_client: List[List[Union[Any]]], num_bins: int = 0
) -> List[List[float]]:
    """Get the distributions (percentages) for multiple clients' targets.

    Parameters
    ----------
    targets_per_client : list of lists, array-like
        Targets (labels) for each client (local node).
    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    distributions: list of lists, array like
        Distributions (percentages) of the clients' targets.
    """
    # Flatten targets array
    targets = np.concatenate(targets_per_client)

    # Bin target for regression tasks
    if num_bins > 0:
        targets_per_client, targets = bin_targets_per_client(
            targets, targets_per_client, num_bins
        )

    # Get unique classes and counts
    unique_classes, _ = np.unique(targets, return_counts=True)

    # Calculate distribution (percentage) for each client
    distributions = []
    for client_targets in targets_per_client:
        # Count occurrences of each unique class in client's targets
        client_counts = np.bincount(
            np.searchsorted(unique_classes, client_targets),
            minlength=len(unique_classes),
        )
        # Get percentages
        client_percentage = client_counts / len(client_targets)
        distributions.append(client_percentage.tolist())

    return distributions


def bin_targets(
    targets: Union[np.ndarray[Any, np.dtype[Any]], List[Any]], num_bins: int
) -> Tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Get the target binned.

    Parameters
    ----------
    targets : lists
        Targets (labels) variable.

    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    bins: list
        Bins calculated.
    binned_targets:
        Binned target variable.
    """
    # Compute bins
    bins = np.linspace(min(targets), max(targets), num_bins + 1)
    # Bin the targets
    binned_targets = np.digitize(targets, bins)
    return bins, binned_targets


def bin_targets_per_client(
    targets: Union[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[np.int64]]],
    targets_per_client: Union[List[List[Union[int, str, bool]]], List[List[int]]],
    num_bins: int,
) -> Tuple[List[List[Any]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Get the target binned.

    Parameters
    ----------
    targets : lists
        Targets (labels) variable.
    targets_per_client : lists of lists, array-like
        Targets (labels) for each client (local node).
    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    binned_targets_per_client: list
        Bins calculated target of each client.
    binned_targets:
        Binned target variable.
    """
    # Bin targets
    bins, binned_targets = bin_targets(targets, num_bins)
    # Bin each clients' target using calculated bins
    binned_targets_per_client = []
    for client_targets in targets_per_client:
        binned_client_targets = list(np.digitize(np.array(client_targets), bins))
        binned_targets_per_client.append(binned_client_targets)
    return binned_targets_per_client, binned_targets


def is_type(lst: List[List[Union[Any]]], data_type: Any) -> bool:
    """Check if values of lists are of certain type.

    Parameters
    ----------
    lst : list of lists, array-like
        Targets (labels) for each client (local node).
    data_type : Python data type
        Desired data type to check
    """
    if data_type == int:
        return any(isinstance(item, int) for sublist in lst for item in sublist)
    elif data_type == float:
        return any(isinstance(item, float) for sublist in lst for item in sublist)
    elif data_type == str:
        return any(isinstance(item, str) for sublist in lst for item in sublist)
    elif data_type == bool:
        return any(isinstance(item, bool) for sublist in lst for item in sublist)
    else:
        raise ValueError(
            "Unsupported data type. Please choose from int, float, str, or bool."
        )


def entropy(
    distribution: Union[np.ndarray[Any, np.dtype[Any]], List[float]],
    normalize: bool = True,
) -> Any:
    """Calculate the entropy.

    Parameters
    ----------
    distribution : list of lists, array-like
        Distribution (percentages) of targets for each local node (client).
    normalize : bool
        Flag to normalize the entropy.

    Returns
    -------
    entropy_val: float
        Entropy.
    """
    entropy_value = -sum(p * math.log2(p) for p in distribution if p != 0)
    if normalize:
        max_entropy = math.log2(np.array(distribution).shape[0])
        return entropy_value / max_entropy
    return entropy_value


def normalize_value(value: float, min_value: float = 0, max_value: float = 1) -> float:
    """Scale (Normalize) input value between min_val and max_val.

    Parameters
    ----------
    value : float
        Value to be normalized.
    min_value : float
        Minimum bound of normalization.
    max_value : float
        Maximum bound of normalization.

    Returns
    -------
    value_normalized: float
        Normalized value between min_val and max_val.
    """
    value_normalized = (value - min_value) / (max_value - min_value)
    return value_normalized
