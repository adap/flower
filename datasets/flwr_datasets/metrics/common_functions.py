# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Common functions in Flower Datasets Metrics."""

import math
from typing import List, Tuple, Type, Union

import numpy as np


def get_distros(
    targets_per_client: List[List[Union[int, float, str]]], num_bins: int = 0
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
    unique_classes, counts = np.unique(targets, return_counts=True)

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
    targets: List[Union[float]], num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    targets: List[Union[float]],
    targets_per_client: List[List[Union[int, float]]],
    num_bins: int,
) -> Tuple[List[List[int]], np.ndarray]:
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
        binned_client_targets = list(np.digitize(client_targets, bins))
        binned_targets_per_client.append(binned_client_targets)
    return binned_targets_per_client, binned_targets


def is_type(lst: List[List[Union[int, float, str]]], data_type: Type) -> bool:
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


def entropy(distribution: np.ndarray, normalize: bool = True) -> float:
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
    entropy_value = -sum([p * math.log2(p) for p in distribution if p != 0])
    if normalize:
        max_entropy = math.log2(distribution.shape[0])
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
