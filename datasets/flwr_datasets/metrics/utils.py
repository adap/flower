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


from typing import List, Union

import pandas as pd


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
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros


def compute_distribution(
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
