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
"""Utils for partitioners."""


import json
from typing import Dict, List


def load_partition_id_to_indices(path: str) -> Dict[int, List[int]]:
    """Load JSON file that contains partition id as keys and list of indices as values.

    Parameters
    ----------
    path: str
        Path to the partition_id_to_indices JSON file.

    Returns
    -------
    partition_id_to_indices: Dict[int, List[int]]
        Partition id to indices mapping representing the result of partitioning.
    """
    with open(path) as indices_mapping_file:
        str_partition_id_to_indices = json.load(indices_mapping_file)
        # Convert keys, which are always strings to ints
        partition_id_to_indices: Dict[int, List[int]] = {
            int(partition_id): indices_list
            for partition_id, indices_list in str_partition_id_to_indices.items()
        }
    return partition_id_to_indices
