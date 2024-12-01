# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
# Copyright 2024 zk0 DBA. All Rights Reserved.
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
"""FederatedLeRobotDataset."""



from flwr_datasets import FederatedDataset



class FederatedLeRobotDataset(FederatedDataset):
    def _prepare_dataset(self) -> None:
        """Overrides the implementation of FederatedDataset in order
        to account for LeRobot specific dataset loading.

        Prepare the dataset (prior to partitioning) by downloading.

        Currently only "train" dataset is loaded from LeRobot datasets. Test datasets are generated via gym.

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.

        NOTE:
        Currently the following two parameters from FederatedDataset are not supported:
        subset - dataset subset loading not supported.
        shuffle - dataset shuffling not supported in this type of dataset.
        """
        self._dataset = {
            "train": {"dataset_name": self._dataset_name, **self._load_dataset_kwargs}
        }
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = {split: False for split in available_splits}
        self._dataset_prepared = True
