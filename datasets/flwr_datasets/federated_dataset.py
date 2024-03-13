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
"""FederatedDataset."""


from typing import Dict, List, Optional, Tuple, Union, cast

import datasets
from datasets import Dataset, DatasetDict
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from flwr_datasets.utils import (
    _check_if_dataset_tested,
    _instantiate_partitioners,
    _instantiate_resplitter_if_needed,
    divide_dataset,
)


# flake8: noqa: E501
# pylint: disable=line-too-long
class FederatedDataset:
    """Representation of a dataset for federated learning/evaluation/analytics.

    Download, partition data among clients (edge devices), or load full dataset.

    Partitions are created using IidPartitioner. Support for different partitioners
    specification and types will come in future releases.

    Parameters
    ----------
    dataset : str
        The name of the dataset in the Hugging Face Hub.
    subset : str
        Secondary information regarding the dataset, most often subset or version
        (that is passed to the name in datasets.load_dataset).
    resplitter : Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]]
        `Callable` that transforms `DatasetDict` splits, or configuration dict for
        `MergeResplitter`.
    partitioners : Dict[str, Union[Partitioner, int]]
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be partitioned
        into). One or multiple `Partitioner` objects can be specified in that manner,
        but at most, one per split.
    partition_division : Optional[Union[List[float], Tuple[float, ...],
    Dict[str, float], Dict[str, Optional[Union[List[float], Tuple[float, ...],
    Dict[str, float]]]]]]
        Fractions specifing the division of the partition assiciated with certain split
        (and partitioner) that enable returning already divided partition from the
        `load_partition` method. You can think of this as on-edge division of the data
        into multiple divisions (e.g. into train and validation). You can also name the
        divisions by using the Dict or create specify it as a List/Tuple. If you
        specified a single partitioner you can provide the simplified form e.g.
        [0.8, 0.2] or {"partition_train": 0.8, "partition_test": 0.2} but when multiple
        partitioners are specified you need to indicate the result of which partitioner
        are further divided e.g. {"train": [0.8, 0.2]} would result in dividing only the
        partitions that are created from the "train" split.
    shuffle : bool
        Whether to randomize the order of samples. Applied prior to resplitting,
        speratelly to each of the present splits in the dataset. It uses the `seed`
        argument. Defaults to True.
    seed : Optional[int]
        Seed used for dataset shuffling. It has no effect if `shuffle` is False. The
        seed cannot be set in the later stages.

    Examples
    --------
    Use MNIST dataset for Federated Learning with 100 clients (edge devices):

    >>> mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> # Load partition for client with ID 10.
    >>> partition = mnist_fds.load_partition(10, "train")
    >>> # Use test split for centralized evaluation.
    >>> centralized = mnist_fds.load_split("test")

    Automatically divde the data returned from `load_partition`
    >>> mnist_fds = FederatedDataset(
    >>>     dataset="mnist",
    >>>     partitioners={"train": 100},
    >>>     partition_division=[0.8, 0.2],
    >>> )
    >>> partition_train, partition_test = mnist_fds.load_partition(10, "train")
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        dataset: str,
        subset: Optional[str] = None,
        resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,
        partitioners: Dict[str, Union[Partitioner, int]],
        partition_division: Optional[
            Union[
                List[float],
                Tuple[float, ...],
                Dict[str, float],
                Dict[
                    str,
                    Optional[Union[List[float], Tuple[float, ...], Dict[str, float]]],
                ],
            ]
        ] = None,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        _check_if_dataset_tested(dataset)
        self._dataset_name: str = dataset
        self._subset: Optional[str] = subset
        self._resplitter: Optional[Resplitter] = _instantiate_resplitter_if_needed(
            resplitter
        )
        self._partitioners: Dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )
        self._partition_division = self._initialize_partition_division(
            partition_division
        )
        self._shuffle = shuffle
        self._seed = seed
        #  _dataset is prepared lazily on the first call to `load_partition`
        #  or `load_split`. See _prepare_datasets for more details
        self._dataset: Optional[DatasetDict] = None
        # Indicate if the dataset is prepared for `load_partition` or `load_split`
        self._dataset_prepared: bool = False

    def load_partition(
        self,
        partition_id: int,
        split: Optional[str] = None,
    ) -> Union[Dataset, List[Dataset], DatasetDict]:
        """Load the partition specified by the idx in the selected split.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        partition_id : int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split : Optional[str]
            Name of the (partitioned) split (e.g. "train", "test"). You can skip this
            parameter if there is only one partitioner for the dataset. The name will be
            inferred automatically. For example, if `partitioners={"train": 10}`, you do
            not need to provide this argument, but if `partitioners={"train": 10,
            "test": 100}`, you need to set it to differentiate which partitioner should
            be used.

        Returns
        -------
        partition : Union[Dataset, List[Dataset], DatasetDict]
            Undivided or divided partition from the dataset split.
            If `partition_division` is not specified then `Dataset` is returned.
            If `partition_division` is specified as `List` or `Tuple` then
            `List[Dataset]` is returned.
            If `partition_division` is specified as `Dict` then `DatasetDict` is
            returned.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        partition = partitioner.load_partition(partition_id)
        if self._partition_division is None:
            return partition
        partition_division = self._partition_division.get(split)
        if partition_division is None:
            return partition
        divided_partition: Union[List[Dataset], DatasetDict] = divide_dataset(
            partition, partition_division
        )
        return divided_partition

    def load_split(self, split: str) -> Dataset:
        """Load the full split of the dataset.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        split : str
            Split name of the downloaded dataset (e.g. "train", "test").

        Returns
        -------
        dataset_split : Dataset
            Part of the dataset identified by its split name.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        self._check_if_split_present(split)
        return self._dataset[split]

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split (for partitioning or full return) is in the dataset."""
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        """Check if the split has corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to perform "
                f"partitioning. Partitioners were specified for the following splits:"
                f"'{partitioners_keys}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        """Assign the corresponding split of the dataset to the partitioner.

        Assign only if the dataset is not assigned yet.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if not self._partitioners[split].is_dataset_assigned():
            self._partitioners[split].dataset = self._dataset[split]

    def _prepare_dataset(self) -> None:
        """Prepare the dataset (prior to partitioning) by download, shuffle, replit.

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.

        Notes
        -----
        The shuffling should happen before the resplitting. Here is the explanation.
        If the dataset has a non-random order of samples e.g. each split has first
        only label 0, then only label 1. Then in case of resplitting e.g.
        someone creates: "train" train[:int(0.75 * len(train))], test: concat(
        train[int(0.75 * len(train)):], test). The new test took the 0.25 of e.g.
        the train that is only label 0 (assuming the equal count of labels).
        Therefore, for such edge cases (for which we have split) the split should
        happen before the resplitting.
        """
        self._dataset = datasets.load_dataset(
            path=self._dataset_name, name=self._subset
        )
        if self._shuffle:
            # Note it shuffles all the splits. The self._dataset is DatasetDict
            # so e.g. {"train": train_data, "test": test_data}. All splits get shuffled.
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._resplitter:
            self._dataset = self._resplitter(self._dataset)
        self._dataset_prepared = True

    def _check_if_no_split_keyword_possible(self) -> None:
        if len(self._partitioners) != 1:
            raise ValueError(
                "Please set the `split` argument. You can only omit the split keyword "
                "if there is exactly one partitioner specified."
            )

    def _initialize_partition_division(
        self,
        partition_division: Optional[
            Union[
                List[float],
                Tuple[float, ...],
                Dict[str, float],
                Dict[
                    str,
                    Optional[Union[List[float], Tuple[float, ...], Dict[str, float]]],
                ],
            ]
        ],
    ) -> Optional[
        Dict[
            str,
            Optional[Union[List[float], Tuple[float, ...], Dict[str, float]]],
        ]
    ]:
        """Create the partition division in the full format.

        Reduced format (possible if only one partitioner exist):

        Union[List[float], Tuple[float, ...], Dict[str, float]

        Full format: Dict[str, Reduced format]
        Full format represents the split to division mapping.
        """
        # Check for simple dict, list, or tuple types directly
        if isinstance(partition_division, (list, tuple)) or (
            isinstance(partition_division, dict)
            and all(isinstance(value, float) for value in partition_division.values())
        ):
            if len(self._partitioners) > 1:
                raise ValueError(
                    f"The specified partition_division {partition_division} does not "
                    f"provide mapping to split but more than one partitioners is "
                    f"specified. Please adjust the partition_division specification to "
                    f"have the split names as the keys."
                )
            return cast(
                Dict[
                    str,
                    Optional[Union[List[float], Tuple[float, ...], Dict[str, float]]],
                ],
                {list(self._partitioners.keys())[0]: partition_division},
            )
        if isinstance(partition_division, dict):
            return cast(
                Dict[
                    str,
                    Optional[Union[List[float], Tuple[float, ...], Dict[str, float]]],
                ],
                partition_division,
            )
        if partition_division is None:
            return None
        raise TypeError("Unsupported type for partition_division")
