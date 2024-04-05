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
"""HellingerDistanceClassification function to calculate Hellinger distance for
classification tasks.
"""

from typing import List, Union

import numpy as np

from flwr_datasets.metrics.common_functions import get_distros, is_type


def HellingerDistanceClassification(
    targets_per_client: List[List[Union[int, str, bool]]]
) -> float:
    """Calculate the Hellinger distance for multiple clients' targets (classification
    tasks).

    Parameters
    ----------
    targets_per_client : list of lists, array-like
        Targets (labels) for each client (local node).

    Returns
    -------
    hellinger_dist: float
        Hellinger distance.

    Examples
    --------
    import numpy as np

    from flwr_datasets.metrics.hellinger_distance_classification import HellingerDistanceClassification

    random_targets_lists = np.random.randint(low=0, high=20, size=(10, 100)).tolist()

    HD = HellingerDistanceClassification(random_targets_lists)

    print(HD)
    """
    # Check if the target values match the task
    if (
        is_type(targets_per_client, int)
        or is_type(targets_per_client, str)
        or is_type(targets_per_client, bool)
    ):
        # Get distributions for clients' targets
        distributions = get_distros(targets_per_client)

        n = len(distributions)
        sqrt_distro = np.sqrt(distributions)
        hellinger_dist = np.sum(
            (sqrt_distro[:, np.newaxis, :] - sqrt_distro[np.newaxis, :, :]) ** 2, axis=2
        )
        hellinger_dist = np.sqrt(np.sum(hellinger_dist) / (2 * n * (n - 1)))
        hellinger_dist = min(hellinger_dist, 1.0)
    else:
        raise ValueError(
            "Unsupported data type for classification task. They must be int, str, or bool."
        )

    return hellinger_dist
