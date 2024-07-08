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
"""DataConnector."""


from typing import Any, Dict, Optional


class DataConnector:
    data: Dict[str, Any]

    def __init__(self, dataset_dict: Optional[Dict[str, Any]]) -> None:
        """Pass datasets/dataloaders ready to be used."""
        self.data = dataset_dict


class DataConnectorWithPartitioning(DataConnector):

    num_partitions: int
