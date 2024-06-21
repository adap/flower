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


import inspect
import json
from typing import Any, Dict, List


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


def _extract_private_attributes_from_object(instance: object) -> Dict[str, Any]:
    """Extract all the private attributes based on the constructor parameters.

    It is assumed that all parameters are stored as attributes.
    e.g. in IidPartitioner(num_partitions) it is assumed that there exist
    _num_partitions. This method extracts all the attributes that should exist.
    In this way for IidPartitioner the following dict is returned
    {"num_partitions": instance._num_partitions}.
    It is a utility function for to_config saving.

    Parameters
    ----------
    instance: object
        object from which the private attributes will be extracted

    Returns
    -------
    name_in_constructor_to_private_attribute: Dict[str, Any]
        Dictionary from that maps names in constructor to saved private attribute values
    """
    constructor_parameters = [
        p for p in inspect.signature(instance.__init__).parameters if p != "self"
    ]

    name_in_constructor_to_private_attribute = {}
    for constructor_parameter in constructor_parameters:
        # add the leading underscore
        private_attr = f"_{constructor_parameter}"
        if hasattr(instance, private_attr):
            name_in_constructor_to_private_attribute[constructor_parameter] = getattr(
                instance, private_attr
            )
    return name_in_constructor_to_private_attribute

def _remove_leading_underscores_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all the keys from the dictionary that start with underscore (_).
    """
    new_config = {}
    for key, value in config.items():
        if not key.startswith("_"):
            new_config[key] = value

    return new_config
