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
"""JensenShannonDistanceRegression function to calculate Jensen-Shannon distance for
regression tasks.
"""

import math
from typing import List, Union

import numpy as np

from flwr_datasets.metrics.common_functions import (
    entropy,
    get_distros,
    is_type,
    normalize_value,
)


def JensenShannonDistanceRegression(
    targets_per_client: List[List[Union[int, float]]], num_bins: int = 20
) -> float:
    """Calculate the Jensen-Shannon distance for multiple numerical clients' targets
    (i.e., regression tasks).

    Parameters
    ----------
    targets_per_client : list of lists, array-like
        Targets (labels) for each client (local node).
    num_bins : int
        Number of bins used to bin the targets for numerical targets (i.e., regression tasks).

    Returns
    -------
    jensen_shannon_dist: float
        Jensen-Shannon distance.

    Examples
    --------
    import numpy as np

    from flower.datasets.flwr_datasets.metrics.jensen_shannon_distance_regression import JensenShannonDistanceRegression

    random_targets_lists = np.random.randint(low=0, high=20, size=(10, 100)).tolist()

    JSD = JensenShannonDistanceRegression(random_targets_lists, num_bins=10)

    print(JSD)
    """
    # Check if the target values match the task
    if is_type(targets_per_client, int) or is_type(targets_per_client, float):
        # Get distributions for clients' targets
        distributions = get_distros(targets_per_client, num_bins)

        # Set weights to be uniform
        weight = 1 / len(distributions)
        jensen_shannon_left = np.zeros(len(distributions[0]))
        jensen_shannon_right = 0
        for distro in distributions:
            jensen_shannon_left += np.array(distro) * weight
            jensen_shannon_right += weight * entropy(distro, normalize=False)

        jensen_shannon_dist = (
            entropy(jensen_shannon_left, normalize=False) - jensen_shannon_right
        )

        if len(distributions) > 2:
            jensen_shannon_dist = normalize_value(
                jensen_shannon_dist,
                min_value=0,
                max_value=math.log2(len(distributions)),
            )

        jensen_shannon_dist = min(np.sqrt(jensen_shannon_dist), 1.0)
    else:
        raise ValueError(
            "Unsupported data type for classification task. They must be int, or float."
        )

    return jensen_shannon_dist
